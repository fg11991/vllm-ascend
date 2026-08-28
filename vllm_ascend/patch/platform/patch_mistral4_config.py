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

import json
from pathlib import Path

from vllm.config.model import ModelConfig
from vllm.logger import logger
from vllm.platforms import current_platform

from vllm_ascend.quantization import utils as quantization_utils

_original_verify_quantization = getattr(ModelConfig, "_verify_quantization", None)
_original_auto_detect_quantization = quantization_utils.maybe_auto_detect_quantization

_MISTRAL_NATIVE_EAGLE_ARCHITECTURE = "EagleMistralLarge3ForCausalLM"


def _is_mistral4_config(config: ModelConfig) -> bool:
    hf_config = getattr(config, "hf_config", None)
    candidates = (
        getattr(config, "model_arch_config", None),
        getattr(config, "hf_text_config", None),
        hf_config,
        getattr(hf_config, "text_config", None),
    )
    return any(
        getattr(candidate, "model_type", None) == "mistral4" for candidate in candidates if candidate is not None
    )


def _is_mistral_native_fp8_eagle_config(config: ModelConfig) -> bool:
    hf_config = getattr(config, "hf_config", None)
    model_arch_config = getattr(config, "model_arch_config", None)
    architectures = getattr(model_arch_config, "architectures", None)
    if architectures is None:
        architectures = getattr(hf_config, "architectures", None)
    if not architectures or _MISTRAL_NATIVE_EAGLE_ARCHITECTURE not in architectures:
        return False

    model_type = getattr(model_arch_config, "model_type", None)
    if model_type is None:
        model_type = getattr(hf_config, "model_type", None)
    if model_type not in ("deepseek_mtp", "eagle"):
        return False

    model = getattr(config, "model", None)
    if not isinstance(model, str):
        return False
    params_path = Path(model) / "params.json"
    if not params_path.is_file():
        return False

    try:
        with params_path.open(encoding="utf-8") as params_file:
            params = json.load(params_file)
    except (OSError, TypeError, ValueError):
        return False

    quantization = params.get("quantization")
    return (
        isinstance(quantization, dict)
        and str(quantization.get("qformat_weight", "")).startswith("fp8_")
        and quantization.get("qscheme_act") == "TENSOR"
    )


def _needs_mistral_fp8_dequantization(config: ModelConfig) -> bool:
    return _is_mistral4_config(config) or _is_mistral_native_fp8_eagle_config(config)


def _checkpoint_label(config: ModelConfig) -> str:
    if _is_mistral_native_fp8_eagle_config(config):
        return "Mistral native EAGLE"
    return "Mistral 4"


def _should_dequantize(config: ModelConfig, quant_method: str | None) -> bool:
    return current_platform.device_name == "npu" and _needs_mistral_fp8_dequantization(config) and quant_method == "fp8"


def _is_fp8_checkpoint(config: ModelConfig) -> bool:
    if not _needs_mistral_fp8_dequantization(config):
        return False

    model_arch_config = getattr(config, "model_arch_config", None)
    hf_config = getattr(config, "hf_config", None)
    quant_config = getattr(model_arch_config, "quantization_config", None)
    if quant_config is None:
        quant_config = getattr(hf_config, "quantization_config", None)

    if isinstance(quant_config, dict):
        quant_method = quant_config.get("quant_method")
    else:
        quant_method = getattr(quant_config, "quant_method", None)
    if quant_method is None:
        model = getattr(config, "model", None)
        if model is None:
            return False
        quant_method = quantization_utils.detect_quantization_method(
            model,
            revision=getattr(config, "revision", None),
        )
    return quant_method == "fp8"


def _patched_verify_quantization(self: ModelConfig) -> None:
    assert _original_verify_quantization is not None

    original_platform_verify = getattr(current_platform, "verify_quantization", None)
    should_dequantize = current_platform.device_name == "npu" and _needs_mistral_fp8_dequantization(self)

    def _platform_verify_hook(quant_method: str | None) -> None:
        if _should_dequantize(self, quant_method):
            logger.info(
                "Detected a per-tensor fp8 %s checkpoint on NPU. "
                "Loading strictly paired, dequantized bf16 weights instead.",
                _checkpoint_label(self),
            )
            self.quantization = None
            return
        assert original_platform_verify is not None
        return original_platform_verify(quant_method)

    try:
        if original_platform_verify is not None:
            current_platform.verify_quantization = _platform_verify_hook
        return _original_verify_quantization(self)
    finally:
        if original_platform_verify is not None:
            current_platform.verify_quantization = original_platform_verify
        if should_dequantize and getattr(self, "quantization", None) == "fp8":
            self.quantization = None


def _patched_auto_detect_quantization(vllm_config) -> None:
    model_config = vllm_config.model_config
    if _is_fp8_checkpoint(model_config):
        model_config.quantization = None
        vllm_config.quant_config = None
        logger.info(
            "Keeping %s fp8 checkpoint quantization disabled on NPU for load-time bf16 dequantization.",
            _checkpoint_label(model_config),
        )
        return
    return _original_auto_detect_quantization(vllm_config)


if _original_verify_quantization is not None:
    ModelConfig._verify_quantization = _patched_verify_quantization

quantization_utils.maybe_auto_detect_quantization = _patched_auto_detect_quantization
