# SPDX-License-Identifier: Apache-2.0
"""CPU coverage of the real v0.30 metrics and typed MsgPack/API boundary."""

import ast
import dataclasses
import importlib.util
import json
import os
import runpy
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import msgspec
import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter

REPO = Path(__file__).resolve().parents[3]
PATCH = REPO / "vllm_ascend/patch/platform/patch_opd_trace.py"


def _source_root():
    configured = os.environ.get("DSPARK_VLLM_SOURCE")
    if configured:
        return Path(configured) / "vllm"
    spec = importlib.util.find_spec("vllm")
    if spec and spec.origin:
        return Path(spec.origin).parent
    pytest.skip("set DSPARK_VLLM_SOURCE to the vLLM v0.30.0 source root")


def _classes(file, names, namespace):
    nodes = [node for node in ast.parse(file.read_text()).body if isinstance(node, ast.ClassDef) and node.name in names]
    assert len(nodes) == len(names)
    # The source is the configured vLLM tree; no models or accelerator imports.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(file), "exec"), namespace)


def _upstream_types():
    root = _source_root()
    ns = dict(dataclass=dataclasses.dataclass, field=dataclasses.field, Any=Any, __name__=__name__)
    _classes(root / "v1/metrics/stats.py", {"RequestSpecDecodeMetrics"}, ns)
    metrics = ns["RequestSpecDecodeMetrics"]
    # Defaults are None for the unrelated payloads. Preserve the actual engine
    # structs and their RequestSpecDecodeMetrics annotation for the wire test.
    ns.update(msgspec=msgspec, Mapping=Mapping, time=time)
    for name in (
        "LogprobsLists",
        "LogprobsTensors",
        "SamplingMaskLists",
        "FinishReason",
        "EngineCoreEvent",
        "PrefillStats",
        "SchedulerStats",
        "UtilityResult",
    ):
        ns[name] = Any
    ns["torch"] = SimpleNamespace(Tensor=Any)
    ns["np"] = SimpleNamespace(ndarray=Any)
    _classes(root / "v1/engine/__init__.py", {"EngineCoreOutput", "UtilityOutput", "EngineCoreOutputs"}, ns)
    return metrics, ns["EngineCoreOutput"], ns["EngineCoreOutputs"]


def _install(metrics):
    with patch.dict(sys.modules, {"vllm.v1.metrics.stats": SimpleNamespace(RequestSpecDecodeMetrics=metrics)}):
        return runpy.run_path(str(PATCH))


def test_real_metrics_typed_codec_and_openai_extra(tmp_path):
    metrics, output_type, outputs_type = _upstream_types()
    unpatched = metrics.new(7)
    unpatched.observe(7, 2, detailed=True)
    original = unpatched.to_dict()
    installed = _install(metrics)
    init, to_dict = metrics.__init__, metrics.to_dict
    installed["_apply_patch"]()
    assert metrics.__init__ is init and metrics.to_dict is to_dict
    assert len([f for f in dataclasses.fields(metrics) if f.name == "opd_trace"]) == 1
    ordinary = metrics.new(7)
    ordinary.observe(7, 2, detailed=True)
    assert ordinary.to_dict() == original
    assert ordinary.opd_trace is None and "opd_trace" not in ordinary.to_dict()

    trace = {
        "version": 1,
        "events": [
            {"trace_id": "one", "offset": 0, "drafted": None},
            {"trace_id": "one", "offset": 1, "drafted": 7, "verified_width": 2, "accepted": 2, "stopped": True},
        ],
    }
    ordinary.opd_trace = trace
    copied = dataclasses.replace(ordinary)
    assert copied.opd_trace == trace and copied.to_dict()["opd_trace"] == trace
    first = output_type(request_id="one", new_token_ids=[4], finish_reason=None)
    final = output_type(request_id="one", new_token_ids=[5], finish_reason="stop", spec_decode_metrics=ordinary)
    batch = outputs_type(outputs=[first, final])
    encoded = msgspec.msgpack.encode(batch)
    decoded = msgspec.msgpack.decode(encoded, type=outputs_type)
    assert decoded.outputs[0].spec_decode_metrics is None
    result = decoded.outputs[1].spec_decode_metrics
    assert result.to_dict() == {**original, "opd_trace": trace}

    class OpenAIBaseModel(BaseModel):
        model_config = ConfigDict(extra="allow")

    ns = dict(OpenAIBaseModel=OpenAIBaseModel, __name__=__name__)
    _classes(_source_root() / "entrypoints/generate/base/protocol.py", {"SpeculativeDecodingMetrics"}, ns)
    api = ns["SpeculativeDecodingMetrics"](**result.to_dict())
    assert json.loads(api.model_dump_json())["opd_trace"] == trace
    assert "opd_trace" not in ns["SpeculativeDecodingMetrics"](**original).model_dump()

    # Independent interpreter builds its own typed decoder and installs the same
    # patch, reproducing the frontend/engine process boundary without an NPU.
    wire = tmp_path / "engine.msgpack"
    wire.write_bytes(encoded)
    script = f"""
import runpy, msgspec
m = runpy.run_path({str(Path(__file__).resolve())!r})
metrics, output_type, outputs_type = m['_upstream_types']()
m['_install'](metrics)
result = msgspec.msgpack.decode(open({str(wire)!r}, 'rb').read(), type=outputs_type)
assert result.outputs[1].spec_decode_metrics.opd_trace == {trace!r}
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_openai_xargs_boolean_coercion():
    adapter = TypeAdapter(dict[str, str | int | float | list[str | int | float]])
    for value, enabled in (
        (True, True),
        (1, True),
        (False, False),
        (0, False),
        ("false", False),
        ("1", False),
        (2, False),
        (1.0, False),
    ):
        actual = adapter.validate_python({"opd_trace_response": value})["opd_trace_response"]
        assert (type(actual) in (bool, int) and actual == 1) is enabled


def test_inline_requests_interleave_and_keep_the_final_round(monkeypatch, caplog, tmp_path):
    import logging

    import torch
    from safetensors.torch import save_file

    metrics, _, _ = _upstream_types()
    _install(metrics)
    legacy = runpy.run_path(str(Path(__file__).with_name("test_opd_trace.py")))

    class MultiScheduler(legacy["SchedulerDouble"]):
        def __init__(self, requests):
            super().__init__(requests[0])
            self.requests = {req.request_id: req for req in requests}

        def update_from_output(self, scheduler_output, model_runner_output):
            model_runner_output.final_metrics = {}
            for rid in scheduler_output.num_scheduled_tokens:
                req = self.requests[rid]
                ids, verified, req.stopped = model_runner_output.steps[rid]
                if verified is not None:
                    req.spec_decode_metrics.observe(*verified, detailed=True)
                req.output_token_ids.extend(ids)
                req.num_output_tokens += len(ids)
                if req.stopped:
                    model_runner_output.final_metrics[rid] = req.spec_decode_metrics
                    del self.requests[rid]
            return model_runner_output

    path = REPO / "vllm_ascend/core/opd_trace.py"
    stubs = {
        "vllm_ascend.core.opd_draft_digest": SimpleNamespace(
            **runpy.run_path(str(path.with_name("opd_draft_digest.py")))
        ),
        "vllm.logger": SimpleNamespace(logger=logging.getLogger("vllm")),
        "vllm.v1.core.sched.scheduler": SimpleNamespace(Scheduler=MultiScheduler),
        "vllm.v1.core.sched.async_scheduler": SimpleNamespace(AsyncScheduler=MultiScheduler),
    }
    with patch.dict(sys.modules, stubs):
        cls = runpy.run_path(str(path))["OPDTraceAsyncScheduler"]
    save_file({"x": torch.tensor([1, 2], dtype=torch.uint8)}, tmp_path / "model.safetensors")
    requests = []
    for rid, extra in (
        ("one", {"opd_trace_id": "trace-one", "opd_trace_response": 1}),
        ("two", {"opd_trace_id": "trace-two", "opd_trace_response": True}),
        ("plain", {"opd_trace_response": 1}),
    ):
        req = SimpleNamespace(
            draft_dir=tmp_path,
            request_id=rid,
            sampling_params=SimpleNamespace(extra_args=extra, eos_token_id=8),
            status=SimpleNamespace(name="FINISHED_LENGTH_CAPPED"),
            stop_reason=None,
            stopped=False,
            num_output_tokens=0,
            spec_decode_metrics=metrics.new(7),
            prompt_token_ids=[4, 2],
            output_token_ids=[],
            num_preemptions=0,
        )
        req.is_finished = lambda req=req: req.stopped
        requests.append(req)
    monkeypatch.setenv("SPECFORGE_OPD_DRAFT_REVISION", "revision")
    with caplog.at_level(logging.INFO):
        scheduler = cls(requests)
        scheduler._runner_widths = True
        first = SimpleNamespace(
            steps={"one": ([3], None, False), "two": ([5], None, False), "plain": ([6], None, True)}
        )
        schedule = SimpleNamespace(num_scheduled_tokens={"two": 1, "plain": 1, "one": 1}, num_invalid_spec_tokens=None)
        scheduler.update_from_output(schedule, first)
        last = SimpleNamespace(
            steps={"one": ([17], (7, 0), True), "two": ([11, 12, 13], (7, 2), True)},
            opd_verify_widths={"one": 7, "two": 2},
        )
        schedule.num_scheduled_tokens = {"one": 1, "two": 1}
        scheduler.update_from_output(schedule, last)
    assert scheduler.requests == {}
    one, two, plain = [req.spec_decode_metrics for req in requests]
    assert one.opd_trace is not two.opd_trace
    assert [e["trace_id"] for e in one.opd_trace["events"]] == ["trace-one", "trace-one"]
    assert [e["trace_id"] for e in two.opd_trace["events"]] == ["trace-two", "trace-two"]
    assert [e["offset"] for e in one.opd_trace["events"]] == [0, 1]
    assert [e["offset"] for e in two.opd_trace["events"]] == [0, 1]
    assert last.final_metrics["one"].opd_trace["events"][-1]["token_ids"] == [17]
    assert last.final_metrics["two"].opd_trace["events"][-1]["verified_width"] == 2
    assert last.final_metrics["two"].opd_trace["events"][-1]["stopped"] is True
    assert plain.opd_trace is None and "opd_trace" not in plain.to_dict()
    assert not any("DSPARK_OPD " in record.message for record in caplog.records)
