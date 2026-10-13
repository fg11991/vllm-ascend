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

"""Capture auxiliary hidden states for a dataset of pre-tokenized samples.

Input is JSONL, one sample per line, with the token ids already rendered by the
training side. Only ``input_ids`` is required::

    {"id": "sample-0001", "input_ids": [151644, 872, ...]}
    {"input_ids": [151644, 872, ...]}

``id`` defaults to the line number. Any other field is carried through to the
manifest untouched, so a loss mask or a source tag can ride along.

Token ids are sent verbatim to /v1/completions. The server never renders a
chat template, because the training side computes its loss mask from its own
rendering and the two token sequences must correspond one to one.

Output, under ``--out-dir``:

* ``captures/<id>.safetensors`` - the connector's file, moved to a stable name
* ``manifest.jsonl`` - one line per sample mapping id to file, token count and
  verification result; this is the request-to-file correspondence record

Re-running skips samples already present in the manifest, so an interrupted run
resumes. Pass ``--overwrite`` to recapture everything.

Usage:
    python3 my_docs/scripts/capture_dataset.py \
        --input /opt/w00958190/data/samples.jsonl \
        --out-dir /opt/w00958190/capture_dataset \
        --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_samples(path: Path) -> list[dict]:
    samples = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "input_ids" not in record:
                raise ValueError(f"{path}:{line_number} has no input_ids")
            record.setdefault("id", f"line-{line_number:06d}")
            samples.append(record)
    return samples


def capture_one(url: str, model: str, token_ids: list[int], timeout: float) -> str:
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
        raise AssertionError("response has no prompt_token_ids")
    if list(prompt_token_ids) != token_ids:
        raise AssertionError("prompt_token_ids differ from what was sent")

    kv_transfer_params = payload.get("kv_transfer_params") or {}
    path = kv_transfer_params.get("hidden_states_path")
    if not path:
        raise AssertionError(f"no hidden_states_path in kv_transfer_params: {kv_transfer_params}")
    return path


def load_capture(path: str) -> dict:
    """Read a capture file, waiting on the connector's lock rather than polling."""
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
            load_hidden_states,
        )

        return load_hidden_states(path)
    except ImportError:
        import fcntl

        from safetensors.torch import load_file

        with open(path + ".lock") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_SH)
            return load_file(path, device="cpu")


def process(sample: dict, args, captures_dir: Path) -> dict:
    """Capture one sample, verify it, and move it to a stable filename."""
    import torch

    token_ids = list(sample["input_ids"])
    sample_id = str(sample["id"])
    started = time.monotonic()

    source_path = capture_one(args.url, args.served_model_name, token_ids, args.timeout)
    data = load_capture(source_path)

    hidden_states, saved_ids = data["hidden_states"], data["token_ids"]
    if saved_ids.tolist() != token_ids:
        raise AssertionError("saved token_ids differ from what was sent")
    if hidden_states.shape[0] != len(token_ids):
        raise AssertionError(
            f"T is {hidden_states.shape[0]} but the prompt has {len(token_ids)} tokens"
        )
    if not torch.isfinite(hidden_states.float()).all():
        raise AssertionError("hidden_states contains NaN or Inf")

    target_path = captures_dir / f"{sample_id}.safetensors"
    shutil.move(source_path, target_path)
    lock_path = Path(source_path + ".lock")
    if lock_path.exists():
        lock_path.unlink()

    carried = {k: v for k, v in sample.items() if k not in {"id", "input_ids"}}
    return {
        "id": sample_id,
        "file": str(target_path),
        "num_tokens": len(token_ids),
        "shape": list(hidden_states.shape),
        "dtype": str(hidden_states.dtype),
        "bytes": target_path.stat().st_size,
        "seconds": round(time.monotonic() - started, 3),
        "verified": True,
        **carried,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="JSONL of pre-tokenized samples")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--served-model-name", default="qwen3.8-flash-next")
    parser.add_argument("--url", default="http://127.0.0.1:9811")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="in-flight requests; keep at or below the server's --max-num-seqs",
    )
    parser.add_argument("--max-tokens-per-sample", type=int, default=0, help="skip longer samples; 0 disables")
    parser.add_argument("--limit", type=int, default=0, help="stop after N samples; 0 means all")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--overwrite", action="store_true", help="recapture samples already in the manifest")
    args = parser.parse_args()

    captures_dir = args.out_dir / "captures"
    captures_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.jsonl"

    done: set[str] = set()
    if manifest_path.exists() and not args.overwrite:
        with manifest_path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    done.add(str(json.loads(line)["id"]))
        print(f"manifest already has {len(done)} samples; they will be skipped")
    elif args.overwrite and manifest_path.exists():
        manifest_path.unlink()

    samples = load_samples(args.input)
    print(f"loaded {len(samples)} samples from {args.input}")

    skipped_long = 0
    pending = []
    for sample in samples:
        if str(sample["id"]) in done:
            continue
        if args.max_tokens_per_sample and len(sample["input_ids"]) > args.max_tokens_per_sample:
            skipped_long += 1
            continue
        pending.append(sample)
        if args.limit and len(pending) >= args.limit:
            break
    if skipped_long:
        print(f"skipping {skipped_long} samples longer than {args.max_tokens_per_sample} tokens")
    if not pending:
        print("nothing to do")
        return 0
    print(f"capturing {len(pending)} samples with concurrency {args.concurrency}")

    started = time.monotonic()
    completed = failed = total_tokens = total_bytes = 0

    with manifest_path.open("a") as manifest, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(process, sample, args, captures_dir): sample for sample in pending}
        for future in as_completed(futures):
            sample = futures[future]
            try:
                record = future.result()
            except Exception as error:  # noqa: BLE001 - report and continue
                failed += 1
                print(f"  FAIL {sample['id']}: {type(error).__name__}: {error}", file=sys.stderr)
                continue
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()
            completed += 1
            total_tokens += record["num_tokens"]
            total_bytes += record["bytes"]
            if completed % 10 == 0 or completed == len(pending):
                elapsed = time.monotonic() - started
                print(
                    f"  {completed}/{len(pending)} done, {failed} failed, "
                    f"{total_tokens / elapsed:.0f} token/s, {total_bytes / 1e9:.2f} GB"
                )

    elapsed = time.monotonic() - started
    print(f"\ncaptured {completed} samples, {failed} failed, in {elapsed / 60:.1f} min")
    if completed:
        print(
            f"throughput: {total_tokens / elapsed:.0f} token/s, "
            f"{completed / elapsed * 60:.1f} sample/min, "
            f"{total_bytes / total_tokens / 1024:.1f} KB/token"
        )
        print(f"on disk   : {total_bytes / 1e9:.2f} GB for {total_tokens} tokens")
    print(f"manifest  : {manifest_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
