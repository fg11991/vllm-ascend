# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Scheduler-only retention for the aligned DSpark context window."""

from collections.abc import Callable
from dataclasses import fields, replace
from functools import partial

from torch.nn import Module
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)

from vllm_ascend.core.kv_cache_interface import AscendDSparkSlidingWindowMLASpec
from vllm_ascend.worker.v2.spec_decode.dspark.context_window import DraftContextWindowConfig, read_context_window_config


def retention_window(config: DraftContextWindowConfig, block_length: int, num_speculative_tokens: int) -> int:
    """Return a conservative scheduler window, never an attention mask.

    vLLM v0.30.0 (ced6857): Scheduler._update_after_schedule increments
    num_computed_tokens AND num_in_flight_tokens by the scheduled count.
    KVCacheManager.allocate_slots -> coordinator.remove_skipped_blocks uses
    C = max(0, total_computed_tokens - num_in_flight_tokens), NOT the async
    optimistic count. Scheduler._connector_finished uses that same
    subtraction. update_from_output drains the in-flight count and subtracts
    rejected tokens before the next schedule. Lookahead only increases
    num_tokens_need_slot; it never increases C. Thus C cannot pass the first
    unsettled step's start, even with multiple queued prefill/decode steps.

    Ascend's ops/triton/v2/spec_decode/prepare_dflash_inputs.py sets
    A = min(last_valid_pos + 1 + B, L) - B (clamped to zero by the reader).
    Rejection can retreat last_valid_pos by at most S speculative tokens;
    the max-model-length clamp can retreat A by at most B. Conservatively
    allow C - A <= S + B (the processed-token path normally gives <= B).
    start >= A - W - (align - 1). SlidingWindowManager frees only complete
    blocks below C - R + 1. R = W + align + B + S + 1 therefore puts that
    boundary <= A - W - align, strictly before start. extra_retained_tokens
    remains zero, so merging specs cannot discard a separate safety margin.

    That kernel takes context positions only from this step's target
    query_start_loc range, masks its rejected suffix, and computes absolute
    slots using the original table. The range begins at the current step's
    start, >= C; even a long prefill chunk therefore writes only resident,
    newly allocated positions. _dflash_local_slot additionally maps null
    block IDs to PAD_SLOT_ID. The windowed table is used only for reads.
    """
    return config.context_window + config.context_window_align + block_length + num_speculative_tokens + 1


def windowed_kv_cache_spec(
    vllm_config: VllmConfig, *, original: Callable[[VllmConfig], KVCacheSpec | None], window: int
) -> KVCacheSpec | None:
    spec = original(vllm_config)
    if not isinstance(spec, AttentionSpec):
        return spec
    if isinstance(spec, SlidingWindowSpec):
        # Existing DSV4 cache subclasses already carry their own storage and
        # grouping contract. Extend retention without replacing that class.
        return replace(spec, sliding_window=max(spec.sliding_window, window))
    if isinstance(spec, MLAAttentionSpec):
        values = {
            field.name: getattr(spec, field.name)
            for field in fields(AscendDSparkSlidingWindowMLASpec)
            if hasattr(spec, field.name) and field.name != "sliding_window"
        }
        return AscendDSparkSlidingWindowMLASpec(**(values | {"sliding_window": window}))
    if type(spec) is FullAttentionSpec:
        values = {field.name: getattr(spec, field.name) for field in fields(AttentionSpec)}
        return SlidingWindowSpec(**values, sliding_window=window)
    raise ValueError(f"Unsupported DSpark context-window cache spec: {type(spec).__name__}")


def validate_windowed_retention(vllm_config: VllmConfig) -> None:
    parallel = vllm_config.parallel_config
    if parallel.decode_context_parallel_size != 1 or parallel.prefill_context_parallel_size != 1:
        raise ValueError("DSpark context-window KV retention does not support DCP/PCP context parallelism.")
    if vllm_config.cache_config.enable_prefix_caching:
        raise ValueError("DSpark context-window KV retention requires --no-enable-prefix-caching.")
    if vllm_config.kv_transfer_config is not None:
        raise ValueError("DSpark context-window KV retention does not yet support a PD KV connector.")
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        raise ValueError("DSpark context-window KV retention requires the hybrid KV cache manager.")


def configure_draft_kv_retention(model: Module, vllm_config: VllmConfig, block_length: int) -> None:
    speculative = vllm_config.speculative_config
    config = read_context_window_config(speculative.draft_model_config.hf_config)
    if config is None:
        return
    validate_windowed_retention(vllm_config)
    window = retention_window(config, block_length, speculative.num_speculative_tokens)
    names = model.get_draft_kv_cache_layer_names()
    layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase, names)
    for name in names:
        layer = layers[name]
        # Override only these loaded draft instances. In particular, leave
        # layer.sliding_window and impl.sliding_window alone: those control
        # FA4 selection and FIA's token-wise mask, not scheduler residency.
        layer.get_kv_cache_spec = partial(windowed_kv_cache_spec, original=layer.get_kv_cache_spec, window=window)
