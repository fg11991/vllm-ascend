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

"""The multi-group KV budget exemption for extract_hidden_states capture.

A packed page-strided backing serves every block-outermost group from one
allocation, so the planner should keep the profiler's full budget instead of
scaling it once per group. A KV transfer normally disqualifies that, because a
connector may own per-layer buffers; the extract_hidden_states connector does
not, so it must keep the exemption.
"""

from types import SimpleNamespace

import pytest
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheGroupSpec, MambaSpec

from vllm_ascend.worker.worker import NPUWorker

AVAILABLE = 20 << 30  # 20 GiB


class FakeLayout:
    is_block_outermost = True
    is_layer_compact = False
    is_block_compact = False


class FakeRunner:
    supports_page_strided_shared_kv_backing = True
    supports_standardized_shared_kv_backing = False
    use_sparse = False
    use_compress = False


def make_groups():
    """Two groups whose summed pages exceed the widest one, so scaling applies."""
    attention = SimpleNamespace(
        page_size_bytes=786432,
        model_version=None,
        __class__=AttentionSpec,
    )
    mamba = SimpleNamespace(
        page_size_bytes=786432,
        model_version=None,
        __class__=MambaSpec,
    )
    return [
        SimpleNamespace(layer_names=["a"], kv_cache_spec=attention, __class__=KVCacheGroupSpec),
        SimpleNamespace(layer_names=["m"], kv_cache_spec=mamba, __class__=KVCacheGroupSpec),
    ]


def make_worker(*, method, with_transfer):
    worker = object.__new__(NPUWorker)
    spec_config = SimpleNamespace(method=method) if method is not None else None
    worker.vllm_config = SimpleNamespace(
        speculative_config=spec_config,
        kv_transfer_config=SimpleNamespace() if with_transfer else None,
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: FakeLayout()),
    )
    worker.model_runner = FakeRunner()
    return worker


@pytest.mark.parametrize("method", ["extract_hidden_states"])
def test_capture_keeps_the_full_budget_despite_a_kv_transfer(method, monkeypatch):
    """Without this, the budget halves and caps the servable context."""
    worker = make_worker(method=method, with_transfer=True)
    monkeypatch.setattr(worker, "_apply_kvpp_memory_budget", lambda value: value, raising=False)
    assert worker._scale_kv_cache_memory_for_multi_group(AVAILABLE) == AVAILABLE


def test_plain_inference_keeps_the_full_budget():
    worker = make_worker(method=None, with_transfer=False)
    assert worker._scale_kv_cache_memory_for_multi_group(AVAILABLE) == AVAILABLE


@pytest.mark.parametrize("method", [None, "mtp", "dspark"])
def test_other_kv_transfers_are_still_scaled(method):
    """A real KV-migrating connector keeps the per-group scaling."""
    worker = make_worker(method=method, with_transfer=True)
    assert worker._scale_kv_cache_memory_for_multi_group(AVAILABLE) < AVAILABLE
