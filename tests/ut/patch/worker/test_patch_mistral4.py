from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm.model_executor.models.mistral4")

from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.mistral_large_3_eagle import (
    EagleMistralLarge3ForCausalLM,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_ascend.patch.worker import patch_mistral4
from vllm_ascend.patch.worker.patch_mistral4 import (
    _dequantize_fp8_weight,
    _fp8_dequant_weight_iter,
)


def test_dequantize_scalar_and_expert_scales() -> None:
    dense = torch.tensor([[1.0, -2.0]], dtype=torch.float8_e4m3fn)
    dense_result = _dequantize_fp8_weight(
        dense,
        torch.tensor(0.5, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        dense_result,
        torch.tensor([[0.5, -1.0]], dtype=torch.bfloat16),
    )

    experts = torch.ones((2, 2, 2), dtype=torch.float8_e4m3fn)
    scales = torch.tensor([2.0, 3.0], dtype=torch.bfloat16).view(2, 1, 1)
    expert_result = _dequantize_fp8_weight(experts, scales)
    torch.testing.assert_close(
        expert_result[0],
        torch.full((2, 2), 2.0, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        expert_result[1],
        torch.full((2, 2), 3.0, dtype=torch.bfloat16),
    )


def test_dequant_iterator_pairs_native_eagle_scale_names() -> None:
    converted = list(
        _fp8_dequant_weight_iter(
            [
                ("eagle_linear.qscale_act", torch.tensor(4.0)),
                ("eagle_linear.qscale_weight", torch.tensor(0.5)),
                (
                    "eagle_linear.weight",
                    torch.ones((2, 2), dtype=torch.float8_e4m3fn),
                ),
            ],
            "Mistral native EAGLE",
        )
    )

    assert [name for name, _ in converted] == ["eagle_linear.weight"]
    torch.testing.assert_close(
        converted[0][1],
        torch.full((2, 2), 0.5, dtype=torch.bfloat16),
    )


def test_dequant_iterator_rejects_unpaired_weight() -> None:
    weight = torch.ones((2, 2), dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="Unpaired Mistral 4"):
        list(_fp8_dequant_weight_iter([("layer.weight", weight)]))


def test_mistral_eagle_init_marks_only_256_rank_draft_mla(monkeypatch) -> None:
    draft_attn = patch_mistral4.MLAAttention.__new__(patch_mistral4.MLAAttention)
    torch.nn.Module.__init__(draft_attn)
    draft_attn.kv_lora_rank = 256
    draft_attn.qk_nope_head_dim = 64
    draft_attn.qk_rope_head_dim = 64
    draft_attn.v_head_dim = 128
    draft_attn.impl = SimpleNamespace()

    draft_model = torch.nn.Module()
    draft_model.add_module("draft_attn", draft_attn)
    eagle = SimpleNamespace()

    def original_init(candidate, *, vllm_config, prefix=""):
        candidate.model = draft_model

    monkeypatch.setattr(patch_mistral4, "_original_eagle_init", original_init)
    monkeypatch.setattr(
        patch_mistral4,
        "current_platform",
        SimpleNamespace(device_name="npu"),
    )

    patch_mistral4._patched_eagle_init(eagle, vllm_config=SimpleNamespace())

    assert draft_attn.impl.kv_cache_lora_rank == 512
    assert draft_attn.impl.prefill_qk_head_dim == 128


def test_mistral_eagle_mla_spec_uses_padded_physical_rank(monkeypatch) -> None:
    original_spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=320,
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
    )
    monkeypatch.setattr(
        patch_mistral4,
        "_original_mla_get_kv_cache_spec",
        lambda _self, _config: original_spec,
    )
    draft_attn = SimpleNamespace(
        kv_lora_rank=256,
        qk_rope_head_dim=64,
        impl=SimpleNamespace(kv_cache_lora_rank=512),
    )

    draft_spec = patch_mistral4._patched_mla_get_kv_cache_spec(
        draft_attn,
        SimpleNamespace(),
    )

    assert draft_spec.head_size == 576


def test_eagle_multimodal_embedding_interface_is_complete() -> None:
    assert EagleMistralLarge3ForCausalLM._embed_text_input_ids is SupportsMultiModal._embed_text_input_ids
    assert EagleMistralLarge3ForCausalLM._has_oov_mm_tokens is False
