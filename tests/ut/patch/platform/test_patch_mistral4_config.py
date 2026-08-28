import json
from types import SimpleNamespace

from vllm_ascend.patch.platform import patch_mistral4_config


def _write_native_eagle_params(tmp_path) -> str:
    model_path = tmp_path / "mistral-eagle"
    model_path.mkdir()
    (model_path / "params.json").write_text(
        json.dumps(
            {
                "quantization": {
                    "qformat_weight": "fp8_e4m3",
                    "qscheme_act": "TENSOR",
                }
            }
        ),
        encoding="utf-8",
    )
    return str(model_path)


def test_nested_mistral4_config_is_detected() -> None:
    config = SimpleNamespace(
        model_arch_config=SimpleNamespace(model_type="mistral3"),
        hf_config=SimpleNamespace(model_type="mistral3"),
        hf_text_config=SimpleNamespace(model_type="mistral4"),
    )

    assert patch_mistral4_config._is_mistral4_config(config)


def test_native_fp8_eagle_is_detected(tmp_path) -> None:
    config = SimpleNamespace(
        model=_write_native_eagle_params(tmp_path),
        model_arch_config=SimpleNamespace(
            model_type="deepseek_mtp",
            architectures=["EagleMistralLarge3ForCausalLM"],
        ),
        hf_config=SimpleNamespace(model_type="deepseek_mtp"),
    )

    assert patch_mistral4_config._is_mistral_native_fp8_eagle_config(config)


def test_deepseek_eagle_is_not_detected(tmp_path) -> None:
    config = SimpleNamespace(
        model=_write_native_eagle_params(tmp_path),
        model_arch_config=SimpleNamespace(
            model_type="eagle",
            architectures=["EagleDeepseekV3ForCausalLM"],
        ),
        hf_config=SimpleNamespace(model_type="eagle"),
    )

    assert not patch_mistral4_config._is_mistral_native_fp8_eagle_config(config)


def test_fp8_discovered_during_verification_is_disabled(monkeypatch) -> None:
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="mistral3",
            text_config=SimpleNamespace(model_type="mistral4"),
        ),
        quantization=None,
    )

    def reject_native_fp8(quant_method: str | None) -> None:
        raise AssertionError(f"native fp8 verification called for {quant_method}")

    platform = SimpleNamespace(
        device_name="npu",
        verify_quantization=reject_native_fp8,
    )

    def original_verify(candidate) -> None:
        candidate.quantization = "fp8"
        platform.verify_quantization(candidate.quantization)

    monkeypatch.setattr(patch_mistral4_config, "current_platform", platform)
    monkeypatch.setattr(
        patch_mistral4_config,
        "_original_verify_quantization",
        original_verify,
    )

    patch_mistral4_config._patched_verify_quantization(config)

    assert config.quantization is None
    assert platform.verify_quantization is reject_native_fp8


def test_fp8_auto_detection_is_not_reapplied(monkeypatch) -> None:
    model_config = SimpleNamespace(
        model_arch_config=SimpleNamespace(
            model_type="mistral4",
            quantization_config={"quant_method": "fp8"},
        ),
        hf_config=SimpleNamespace(model_type="mistral4"),
        quantization=None,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        quant_config="stale-fp8-config",
    )

    patch_mistral4_config._patched_auto_detect_quantization(vllm_config)

    assert model_config.quantization is None
    assert vllm_config.quant_config is None
