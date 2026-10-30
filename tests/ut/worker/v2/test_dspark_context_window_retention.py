# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import is_kv_cache_spec_uniform
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager, register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_ascend.attention import fa4_v1
from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder
from vllm_ascend.core.kv_cache_interface import (
    AscendDSparkSlidingWindowMLASpec,
    AscendMLAAttentionSpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _ascend_get_kv_cache_config_from_groups,
    _get_kv_cache_groups_uniform_page_size,
    group_and_unify_kv_cache_specs,
)
from vllm_ascend.worker.v2.spec_decode.dspark import cache_retention
from vllm_ascend.worker.v2.spec_decode.dspark.context_window import DraftContextWindow, DraftContextWindowConfig


def make_config(enabled=True):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=7,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(dflash_config={"context_window": 256, "context_window_align": 128})
                if enabled
                else SimpleNamespace()
            ),
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        kv_transfer_config=None,
    )


def full_spec(spec_type=FullAttentionSpec, **kwargs):
    return spec_type(block_size=128, num_kv_heads=1, head_size=256, dtype=torch.bfloat16, **kwargs)


@pytest.mark.parametrize("spec_type", [FullAttentionSpec, MLAAttentionSpec, AscendMLAAttentionSpec])
@pytest.mark.parametrize("enabled", [False, True])
def test_only_draft_spec_changes(monkeypatch, spec_type, enabled):
    config = make_config(enabled)
    original_spec = full_spec(spec_type)
    original = MagicMock(return_value=original_spec)
    layer = SimpleNamespace(get_kv_cache_spec=original, sliding_window=None, impl=SimpleNamespace(sliding_window=None))
    model = SimpleNamespace(get_draft_kv_cache_layer_names=lambda: ["draft"])
    monkeypatch.setattr(cache_retention, "get_layers_from_vllm_config", lambda config, cls, names: {"draft": layer})
    cache_retention.configure_draft_kv_retention(model, config, 7)
    spec = layer.get_kv_cache_spec(config)
    original.assert_called_once_with(config)
    assert layer.sliding_window is None and layer.impl.sliding_window is None
    assert spec.page_size_bytes == original_spec.page_size_bytes
    if not enabled:
        assert spec is original_spec and layer.get_kv_cache_spec is original
        return
    assert isinstance(spec, SlidingWindowSpec)
    assert spec.sliding_window == 399
    if spec_type is not FullAttentionSpec:
        assert isinstance(spec, AscendDSparkSlidingWindowMLASpec)
        assert not isinstance(spec, FullAttentionSpec)
        assert spec.supports_single_raw_backing
        assert type(spec).merge([spec, spec]) == spec
        register_all_kvcache_specs(None)
        register_ascend_kv_cache_specs()
        assert KVCacheSpecRegistry.get_manager_class(spec) is SlidingWindowManager
        assert group_and_unify_kv_cache_specs({"target": original_spec, "draft": spec}) is None
        assert not is_kv_cache_spec_uniform({"target": original_spec, "draft": spec})
        assert not is_kv_cache_spec_uniform({"draft": spec, "target": original_spec})


@pytest.mark.parametrize("mode", ["dcp", "pcp", "prefix", "pd", "hybrid_disabled"])
def test_unsupported_settings_fail_at_startup(mode):
    config = make_config()
    if mode == "dcp":
        config.parallel_config.decode_context_parallel_size = 2
    elif mode == "pcp":
        config.parallel_config.prefill_context_parallel_size = 2
    elif mode == "prefix":
        config.cache_config.enable_prefix_caching = True
    elif mode == "pd":
        config.kv_transfer_config = object()
    else:
        config.scheduler_config.disable_hybrid_kv_cache_manager = True
    with pytest.raises(ValueError, match="DSpark context-window KV retention"):
        cache_retention.validate_windowed_retention(config)
    # Opt-out keeps existing configurations (including prefix caching/PD) valid.
    config.speculative_config.draft_model_config.hf_config = SimpleNamespace()
    cache_retention.configure_draft_kv_retention(None, config, 7)


def test_non_paged_draft_cache_is_unchanged():
    spec = MambaSpec(block_size=128, shapes=((1,),), dtypes=(torch.float32,))
    assert cache_retention.windowed_kv_cache_spec(None, original=lambda _: spec, window=399) is spec


@pytest.mark.parametrize(
    "window,align,block_length,num_spec,kernel_block,manager_block",
    [
        (64, 16, 1, 1, 16, 16),
        (256, 128, 7, 7, 128, 128),
        (4096, 128, 7, 7, 128, 896),
        (257, 256, 16, 32, 128, 512),
        (1, 32, 3, 8, 16, 128),
    ],
)
@pytest.mark.parametrize("in_flight_steps", [0, 1, 2, 4])
def test_real_manager_never_evicts_read_or_write_blocks(
    window, align, block_length, num_spec, kernel_block, manager_block, in_flight_steps
):
    config = DraftContextWindowConfig(window, align)
    retained = cache_retention.retention_window(config, block_length, num_spec)
    spec = SlidingWindowSpec(
        block_size=manager_block, num_kv_heads=1, head_size=8, dtype=torch.bfloat16, sliding_window=retained
    )
    pool = BlockPool(num_gpu_blocks=4096, enable_caching=False, hash_block_size=manager_block)
    manager = SlidingWindowManager(
        spec, block_pool=pool, enable_caching=False, kv_cache_group_id=0, scheduler_block_size=manager_block
    )
    max_len = 8 * (window + align + num_spec + block_length + manager_block)
    manager.allocate_new_blocks("r", max_len, max_len)
    full_table = torch.zeros((1, (max_len + kernel_block - 1) // kernel_block), dtype=torch.int32)
    reader = DraftContextWindow(
        config,
        group_ids=[0],
        kernel_block_sizes=[kernel_block],
        full_block_tables=[full_table],
        block_length=block_length,
        device=torch.device("cpu"),
    )
    # Sweep every alignment phase, including the largest conservative retreat
    # (all drafts rejected plus the max_model_len clamp).
    for anchor in range(0, max_len - retained, max(1, align // 7)):
        optimistic = anchor + num_spec + block_length + in_flight_steps * (num_spec + 1)
        processed = optimistic - in_flight_steps * (num_spec + 1)
        manager.remove_skipped_blocks("r", processed)
        ids = torch.tensor([block.block_id for block in manager.req_to_blocks["r"]])
        ratio = manager_block // kernel_block
        expanded = ids[:, None] * ratio + torch.arange(ratio)[None, :]
        expanded[ids == 0] = 0
        full_table.copy_(expanded.flatten()[: full_table.shape[1]].unsqueeze(0))
        reader.apply(torch.tensor([anchor + block_length], dtype=torch.int32))
        num_read_blocks = (int(reader.windowed_seq_lens[0]) + kernel_block - 1) // kernel_block
        assert torch.all(reader.block_table_clones[0][0, :num_read_blocks] != 0)
        # Context writes cover only current target positions; the oldest
        # possible one here precedes anchor by the verification block.
        write_start = max(0, anchor - num_spec - 1)
        assert torch.all(
            full_table[0, write_start // kernel_block : (anchor + block_length - 1) // kernel_block + 1] != 0
        )
    assert pool.get_num_free_blocks() > 4096 - 1 - (max_len + manager_block - 1) // manager_block


@pytest.mark.parametrize("min_size", [0, 16])
def test_qwen_hybrid_groups_before_after(min_size):
    # TP4 Qwen GQA page geometry; recurrent payload is padded to this page.
    # This is a grouping fixture, not a real model initialization measurement.
    target = full_spec()
    mamba = MambaSpec(block_size=128, shapes=((1,),), dtypes=(torch.float32,), page_size_padded=target.page_size_bytes)
    specs = {
        **{f"language_model.model.layers.{i}.self_attn.attn": target for i in range(3, 64, 4)},
        **{f"language_model.model.layers.{i}.linear_attn": mamba for i in range(64) if i % 4 != 3},
        **{f"model.layers.{i}.self_attn.attn": target for i in range(5)},
    }
    for enabled in [False, True]:
        current = dict(specs)
        if enabled:
            for name in current:
                if name.startswith("model."):
                    current[name] = cache_retention.windowed_kv_cache_spec(None, original=lambda _: target, window=4239)
        groups = _get_kv_cache_groups_uniform_page_size(current, min_size=min_size)
        planner_config = SimpleNamespace(
            cache_config=SimpleNamespace(
                num_gpu_blocks_override=None,
                prefix_cache_retention_interval=None,
                get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBNHC,
            ),
            attention_config=SimpleNamespace(hisparse_config=None),
            kv_transfer_config=None,
            speculative_config=None,
        )
        planned = _ascend_get_kv_cache_config_from_groups(planner_config, groups, available_memory=1 << 30)
        assert planned.num_blocks > 0
        assert planned.kv_cache_groups == groups
        assert len({tensor.size for tensor in planned.kv_cache_tensors}) == 1
        print(f"\nretention={enabled} min_size={min_size} groups={len(groups)}")
        for group in planned.kv_cache_groups:
            spec = group.kv_cache_spec
            print(type(spec).__name__, spec.block_size, spec.page_size_bytes, group.layer_names)
        layer_specs = [
            inner
            for group in groups
            for inner in (
                group.kv_cache_spec.kv_cache_specs.values()
                if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                else [group.kv_cache_spec]
            )
        ]
        assert {spec.page_size_bytes for spec in layer_specs} == {target.page_size_bytes}
        assert sorted(name for group in groups for name in group.layer_names) == sorted(specs)
        if enabled:
            drafts = [g for g in groups if isinstance(g.kv_cache_spec, SlidingWindowSpec)]
            assert len(drafts) == 1 and len(drafts[0].layer_names) == 5
            assert all(name.startswith("model.") for name in drafts[0].layer_names)
            if min_size == 16:
                assert [len(g.layer_names) for g in groups] == [16, 16, 16, 16, 5]


def test_mla_swa_admission_and_uniformity():
    spec = cache_retention.windowed_kv_cache_spec(
        None, original=lambda _: full_spec(AscendMLAAttentionSpec), window=399
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        model_config=SimpleNamespace(max_model_len=131072),
        max_in_flight_tokens=256,
    )
    assert spec.max_memory_usage_bytes(config) == 7 * spec.page_size_bytes
    assert not spec.is_uniform_with_collection({"other": replace(spec, sliding_window=400)})


def test_chunked_prefill_keeps_every_current_context_write_resident():
    config = DraftContextWindowConfig(256, 128)
    spec = SlidingWindowSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
        sliding_window=cache_retention.retention_window(config, 7, 7),
    )
    pool = BlockPool(num_gpu_blocks=512, enable_caching=False, hash_block_size=128)
    manager = SlidingWindowManager(
        spec, block_pool=pool, enable_caching=False, kv_cache_group_id=0, scheduler_block_size=128
    )
    computed = 0
    for chunk_length in (128, 8192, 129, 4096, 8, 8):
        manager.remove_skipped_blocks("r", computed)
        end = computed + chunk_length
        manager.allocate_new_blocks("r", end + 7, end)
        # Ascend context preparation indexes only this target step's positions,
        # including the whole current chunk even when it is much larger than W.
        blocks = manager.req_to_blocks["r"]
        assert all(blocks[pos // 128].block_id != 0 for pos in range(computed, end + 7))
        if computed > spec.sliding_window + 128:
            assert blocks[0].is_null
        computed = end


@pytest.mark.parametrize("builder_type", [AscendAttentionMetadataBuilder, fa4_v1.AscendFA4MetadataBuilder])
def test_metadata_builder_accepts_retention_spec_without_enabling_a_mask(monkeypatch, builder_type):
    config = MagicMock()
    config.parallel_config.prefill_context_parallel_size = 1
    config.model_config.max_model_len = 32768
    config.model_config.hf_text_config.sliding_window = None
    config.speculative_config.num_speculative_tokens = 7
    config.speculative_config.parallel_drafting = True
    config.scheduler_config.enable_chunked_prefill = True
    monkeypatch.setattr(fa4_v1, "_load_fa4", lambda: MagicMock())
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    original = full_spec()
    retained = cache_retention.windowed_kv_cache_spec(config, original=lambda _: original, window=4239)
    full_builder = builder_type(original, ["draft"], config, torch.device("cpu"))
    swa_builder = builder_type(retained, ["draft"], config, torch.device("cpu"))
    assert full_builder.decode_threshold == swa_builder.decode_threshold == 8
    assert full_builder.max_num_blocks_per_req == swa_builder.max_num_blocks_per_req
    assert config.model_config.hf_text_config.sliding_window is None
    query_start = torch.tensor([0, 7, 14], dtype=torch.int32)
    lengths = torch.tensor([399, 200], dtype=torch.int32)
    table = torch.ones((2, 4), dtype=torch.int32)
    common = SimpleNamespace(causal=False, query_start_loc=query_start)
    before = full_builder._build_fia_seq_inputs(common, 2, query_start, lengths, table)
    after = swa_builder._build_fia_seq_inputs(common, 2, query_start, lengths, table)
    for left, right in zip(before, after):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        else:
            assert left == right
