#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verify the Qwen3.8-Flash-Next auxiliary hidden-state capture end to end.

Sends token ids straight to /v1/completions, as the training side requires: the
loss mask is computed by SpecForge from its own rendering, so the server must
not re-render a chat template. The request API follows speculators'
``src/speculators/data_generation/vllm_client.py``.

Checks performed, matching the capture acceptance list:

1. ``prompt_token_ids`` in the response equals the ids that were sent.
2. The saved ``token_ids`` equals them too, and T equals the prompt length.
3. ``hidden_states`` is ``[T, num_aux, hidden_size]`` in bfloat16.
4. Running the same sample twice yields byte-identical tensors.
5. Throughput and on-disk size, with an extrapolation to a full run.

Not covered here: validating aux id ``num_hidden_layers`` against
``prompt_logprobs``. That needs a second server started without the capture
connector, so it is a separate step.

Usage:
    python3 my_docs/scripts/verify_capture.py \
        --model /opt/model/Qwen3.8-Flash-Next-w8a8-mtp \
        --url http://127.0.0.1:9811 \
        --lengths 100,500,2000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SAMPLE_TEXT = (
    "投机推理通过一个小的草稿模型先行生成若干候选 token，再由目标模型一次性校验，"
    "从而在不改变输出分布的前提下减少目标模型的前向次数。草稿模型的质量决定接受长度，"
    "而接受长度决定端到端加速比。Speculative decoding lets a small draft model propose "
    "several tokens that the target model then verifies in a single forward pass, "
    "reducing the number of target forwards without changing the output distribution. "
)


def build_token_ids(tokenizer, length: int) -> list[int]:
    """Repeat a fixed passage until it reaches exactly ``length`` tokens."""
    ids: list[int] = []
    while len(ids) < length:
        ids.extend(tokenizer(SAMPLE_TEXT, add_special_tokens=False)["input_ids"])
    return ids[:length]


def capture_once(url: str, model: str, token_ids: list[int], timeout: float) -> str:
    """Send one capture request and return the path the connector wrote."""
    import requests

    response = requests.post(
        f"{url}/v1/completions",
        json={
            "model": model,
            "prompt": token_ids,
            "max_tokens": 1,
            "return_token_ids": True,
            "temperature": 0.0,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    prompt_token_ids = payload["choices"][0].get("prompt_token_ids")
    if prompt_token_ids is None:
        raise AssertionError("response has no prompt_token_ids; is return_token_ids honored?")
    if list(prompt_token_ids) != token_ids:
        raise AssertionError(
            f"prompt_token_ids differ from what was sent "
            f"(sent {len(token_ids)} ids, got {len(prompt_token_ids)})"
        )

    kv_transfer_params = payload.get("kv_transfer_params")
    if not kv_transfer_params:
        raise AssertionError(
            "response has no kv_transfer_params; is ExampleHiddenStatesConnector configured?"
        )
    path = kv_transfer_params.get("hidden_states_path")
    if not path:
        raise AssertionError(f"kv_transfer_params has no hidden_states_path: {kv_transfer_params}")
    return path


def load_capture(path: str) -> dict:
    """Read a capture file, waiting on the connector's lock rather than polling."""
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
            load_hidden_states,
        )

        return load_hidden_states(path)
    except ImportError:
        # Fall back to the same flock protocol the connector documents.
        import fcntl

        from safetensors.torch import load_file

        with open(path + ".lock") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_SH)
            return load_file(path, device="cpu")


def check_one(
    tokenizer,
    url: str,
    model: str,
    length: int,
    num_aux: int,
    hidden_size: int,
    timeout: float,
) -> dict:
    import torch

    token_ids = build_token_ids(tokenizer, length)
    print(f"\n--- length {length} ---")

    started = time.monotonic()
    path = capture_once(url, model, token_ids, timeout)
    request_seconds = time.monotonic() - started
    print(f"  prompt_token_ids  : matches ({length} ids)")
    print(f"  hidden_states_path: {path}")

    data = load_capture(path)
    missing = {"hidden_states", "token_ids"} - set(data)
    if missing:
        raise AssertionError(f"capture file is missing {sorted(missing)}; has {sorted(data)}")

    hidden_states, saved_ids = data["hidden_states"], data["token_ids"]
    print(f"  hidden_states     : shape {tuple(hidden_states.shape)} dtype {hidden_states.dtype}")
    print(f"  token_ids         : shape {tuple(saved_ids.shape)} dtype {saved_ids.dtype}")

    if saved_ids.tolist() != token_ids:
        raise AssertionError("saved token_ids differ from the ids that were sent")
    expected_shape = (length, num_aux, hidden_size)
    if tuple(hidden_states.shape) != expected_shape:
        raise AssertionError(f"hidden_states shape is {tuple(hidden_states.shape)}, expected {expected_shape}")
    if hidden_states.dtype != torch.bfloat16:
        print(f"  WARNING           : dtype is {hidden_states.dtype}, expected torch.bfloat16")
    if not torch.isfinite(hidden_states.float()).all():
        raise AssertionError("hidden_states contains NaN or Inf")

    size_bytes = Path(path).stat().st_size
    print(
        f"  OK                : T={length} matches prompt length, "
        f"{size_bytes / 1e6:.1f} MB on disk, request took {request_seconds:.2f}s "
        f"({length / request_seconds:.0f} token/s)"
    )
    return {
        "length": length,
        "path": path,
        "seconds": request_seconds,
        "bytes": size_bytes,
        "hidden_states": hidden_states,
        "token_ids": token_ids,
    }


def check_determinism(tokenizer, url: str, model: str, first: dict, timeout: float) -> None:
    """The same ids must produce byte-identical tensors on a second run."""
    import torch

    print("\n--- determinism (same sample twice) ---")
    path = capture_once(url, model, first["token_ids"], timeout)
    if path == first["path"]:
        print("  note              : connector reused the same filename; comparing in place")
    second = load_capture(path)

    identical = torch.equal(first["hidden_states"], second["hidden_states"])
    print(f"  hidden_states     : identical={identical}")
    if not identical:
        delta = (first["hidden_states"].float() - second["hidden_states"].float()).abs().max()
        raise AssertionError(f"two runs of the same sample differ, max abs diff {delta.item()}")
    if second["token_ids"].tolist() != first["token_ids"]:
        raise AssertionError("token_ids differ between two runs of the same sample")
    print("  OK                : capture is deterministic for this sample")


def report_extrapolation(results: list[dict], samples: int, mean_tokens: int) -> None:
    print("\n--- extrapolation ---")
    total_seconds = sum(r["seconds"] for r in results)
    total_tokens = sum(r["length"] for r in results)
    total_bytes = sum(r["bytes"] for r in results)
    tokens_per_second = total_tokens / total_seconds
    bytes_per_token = total_bytes / total_tokens

    print(f"  measured          : {tokens_per_second:.0f} token/s, {bytes_per_token / 1024:.1f} KB/token")
    print(f"                      (sequential, one request at a time, no concurrency)")
    projected_seconds = samples * mean_tokens / tokens_per_second
    projected_bytes = samples * mean_tokens * bytes_per_token
    print(
        f"  {samples} samples x {mean_tokens} token: "
        f"{projected_seconds / 60:.1f} min, {projected_bytes / 1e9:.1f} GB"
    )
    print("  note              : concurrency was not exercised; a real run should be faster")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="/opt/model/Qwen3.8-Flash-Next-w8a8-mtp")
    parser.add_argument("--served-model-name", default="qwen3.8-flash-next")
    parser.add_argument("--url", default="http://127.0.0.1:9811")
    parser.add_argument("--lengths", default="100,500,2000", help="comma-separated prompt lengths in tokens")
    parser.add_argument("--num-aux", type=int, default=6, help="number of aux ids captured per token")
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--samples", type=int, default=200, help="sample count for the extrapolation")
    parser.add_argument("--mean-tokens", type=int, default=2000, help="mean length for the extrapolation")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    lengths = [int(value) for value in args.lengths.split(",") if value.strip()]

    results = []
    for length in lengths:
        results.append(
            check_one(
                tokenizer,
                args.url,
                args.served_model_name,
                length,
                args.num_aux,
                args.hidden_size,
                args.timeout,
            )
        )

    check_determinism(tokenizer, args.url, args.served_model_name, results[0], args.timeout)
    report_extrapolation(results, args.samples, args.mean_tokens)

    print(f"\nAll checks passed for lengths {lengths}.")
    print("Still to do: validate the last aux id against prompt_logprobs from a")
    print("server started without the capture connector.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
