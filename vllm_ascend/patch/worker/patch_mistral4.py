# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable, Iterator
from dataclasses import replace

import torch
from vllm.logger import logger
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.mistral4 import Mistral4Model
from vllm.model_executor.models.mistral_large_3_eagle import (
    EagleMistralLarge3ForCausalLM,
)
from vllm.platforms import current_platform

FP8_DTYPES = tuple(
    getattr(torch, dtype_name)
    for dtype_name in (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
    )
    if hasattr(torch, dtype_name)
)

MISTRAL_EAGLE_KV_LORA_RANK = 256
MISTRAL_EAGLE_ROPE_HEAD_DIM = 64
MISTRAL_EAGLE_PADDED_KV_LORA_RANK = 512


def _dequantize_fp8_weight(
    fp8_weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
) -> torch.Tensor:
    scale_shape = tuple(weight_scale_inv.shape)
    valid_scale = weight_scale_inv.numel() == 1 or (
        fp8_weight.ndim == 3
        and weight_scale_inv.ndim == 3
        and weight_scale_inv.shape[0] == fp8_weight.shape[0]
        and weight_scale_inv.shape[1:] == (1, 1)
    )
    if not valid_scale:
        raise ValueError(
            f"Unexpected Mistral 4 fp8 inverse-scale shape: weight={tuple(fp8_weight.shape)}, scale={scale_shape}"
        )

    return fp8_weight.to(dtype=torch.bfloat16) * weight_scale_inv.to(dtype=torch.bfloat16)


def _fp8_dequant_weight_iter(
    weights: Iterable[tuple[str, torch.Tensor]],
    checkpoint_label: str = "Mistral 4",
) -> Iterator[tuple[str, torch.Tensor]]:
    pending_weights: dict[str, torch.Tensor] = {}
    pending_scales: dict[str, torch.Tensor] = {}
    converted_tensors = 0
    converted_elements = 0
    skipped_activation_scales = 0

    for name, loaded_weight in weights:
        if name.endswith(("activation_scale", ".input_scale", ".qscale_act")):
            skipped_activation_scales += 1
            continue

        if name.endswith("_scale_inv"):
            weight_name = name[: -len("_scale_inv")]
        elif name.endswith(".weight_scale"):
            weight_name = name[: -len("_scale")]
        elif name.endswith(".qscale_weight"):
            weight_name = name[: -len(".qscale_weight")] + ".weight"
        else:
            weight_name = None

        if weight_name is not None:
            pending_weight = pending_weights.pop(weight_name, None)
            if pending_weight is None:
                pending_scales[name] = loaded_weight
                continue
            loaded_weight = _dequantize_fp8_weight(pending_weight, loaded_weight)
            name = weight_name
            converted_tensors += 1
            converted_elements += loaded_weight.numel()
        elif loaded_weight.dtype in FP8_DTYPES:
            pending_scale = None
            scale_names = [f"{name}_scale_inv", f"{name}_scale"]
            if name.endswith(".weight"):
                scale_names.append(name[: -len(".weight")] + ".qscale_weight")
            for scale_name in scale_names:
                pending_scale = pending_scales.pop(scale_name, None)
                if pending_scale is not None:
                    break
            if pending_scale is None:
                pending_weights[name] = loaded_weight
                continue
            loaded_weight = _dequantize_fp8_weight(loaded_weight, pending_scale)
            converted_tensors += 1
            converted_elements += loaded_weight.numel()

        yield name, loaded_weight

    if pending_weights or pending_scales:
        raise ValueError(
            f"Unpaired {checkpoint_label} fp8 weight/scale tensors: "
            f"pending_weights={sorted(pending_weights)}, "
            f"pending_scales={sorted(pending_scales)}"
        )

    logger.info(
        "Dequantized %d %s fp8 tensors (%d elements) to bf16; ignored %d paired activation-scale metadata tensors.",
        converted_tensors,
        checkpoint_label,
        converted_elements,
        skipped_activation_scales,
    )


_original_load_weights = Mistral4Model.load_weights
_original_mla_get_kv_cache_spec = MLAAttention.get_kv_cache_spec
_original_eagle_init = EagleMistralLarge3ForCausalLM.__init__
_original_eagle_load_weights = EagleMistralLarge3ForCausalLM.load_weights


def _patched_load_weights(
    self: Mistral4Model,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    if current_platform.device_name == "npu":
        weights = _fp8_dequant_weight_iter(weights)
    return _original_load_weights(self, weights)


Mistral4Model.load_weights = _patched_load_weights


def _patched_mla_get_kv_cache_spec(self: MLAAttention, vllm_config):
    spec = _original_mla_get_kv_cache_spec(self, vllm_config)
    cache_rank = getattr(self.impl, "kv_cache_lora_rank", self.kv_lora_rank)
    if cache_rank == self.kv_lora_rank:
        return spec

    expected_head_size = self.kv_lora_rank + self.qk_rope_head_dim
    if spec.head_size != expected_head_size:
        raise ValueError(
            "Unexpected MLA KV cache spec before Mistral EAGLE padding: "
            f"head_size={spec.head_size}, expected={expected_head_size}"
        )
    return replace(spec, head_size=cache_rank + self.qk_rope_head_dim)


MLAAttention.get_kv_cache_spec = _patched_mla_get_kv_cache_spec


def _patched_eagle_init(
    self: EagleMistralLarge3ForCausalLM,
    *,
    vllm_config,
    prefix: str = "",
) -> None:
    _original_eagle_init(self, vllm_config=vllm_config, prefix=prefix)
    if current_platform.device_name != "npu":
        return

    padded_layers = 0
    for module in self.model.modules():
        if not isinstance(module, MLAAttention):
            continue
        if module.kv_lora_rank == MISTRAL_EAGLE_KV_LORA_RANK and module.qk_rope_head_dim == MISTRAL_EAGLE_ROPE_HEAD_DIM:
            module.impl.kv_cache_lora_rank = MISTRAL_EAGLE_PADDED_KV_LORA_RANK
            module.impl.prefill_qk_head_dim = max(
                module.qk_nope_head_dim,
                module.v_head_dim,
            )
            padded_layers += 1

    if padded_layers:
        logger.info(
            "Using a %d-dimension physical MLA KV cache for %d Mistral EAGLE draft layer(s); semantic rank remains %d.",
            MISTRAL_EAGLE_PADDED_KV_LORA_RANK,
            padded_layers,
            MISTRAL_EAGLE_KV_LORA_RANK,
        )


EagleMistralLarge3ForCausalLM.__init__ = _patched_eagle_init


def _patched_eagle_load_weights(
    self: EagleMistralLarge3ForCausalLM,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    if current_platform.device_name == "npu" and self.quant_config is None:
        weights = _fp8_dequant_weight_iter(weights, "Mistral native EAGLE")
    return _original_eagle_load_weights(self, weights)


EagleMistralLarge3ForCausalLM.load_weights = _patched_eagle_load_weights
EagleMistralLarge3ForCausalLM._embed_text_input_ids = (  # type: ignore[attr-defined]
    SupportsMultiModal._embed_text_input_ids
)
EagleMistralLarge3ForCausalLM._has_oov_mm_tokens = False  # type: ignore[attr-defined]
