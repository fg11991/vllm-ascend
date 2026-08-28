# Mistral-Small-4-119B-2603 delivery for vLLM 0.26

This directory contains two independently applicable parts of the Mistral
Small 4 adaptation:

- `vllm-mistral4-v0.26.0.patch` for upstream vLLM.
- `vllm-ascend-mistral4-v0.26.0rc-dspark.patch` for this vLLM-Ascend branch.

## Baseline

- vLLM tag: `v0.26.0`
- vLLM commit: `568afb3a13806beb53bb2e6bd518269357b237c0`
- vLLM-Ascend branch: `v0.26.0rc_dspark_dev`
- vLLM-Ascend commit: `ce7fc11a46479ba753d975eebc1a088d51adb008`

Apply the patch before installing or importing vLLM-Ascend:

```bash
git -C "$VLLM_SRC" apply --check \
  "$VLLM_ASCEND_SRC/patches/mistral4/vllm-mistral4-v0.26.0.patch"
git -C "$VLLM_SRC" apply \
  "$VLLM_ASCEND_SRC/patches/mistral4/vllm-mistral4-v0.26.0.patch"

git -C "$VLLM_ASCEND_SRC" apply --check \
  "$VLLM_ASCEND_SRC/patches/mistral4/vllm-ascend-mistral4-v0.26.0rc-dspark.patch"
git -C "$VLLM_ASCEND_SRC" apply \
  "$VLLM_ASCEND_SRC/patches/mistral4/vllm-ascend-mistral4-v0.26.0rc-dspark.patch"
```

The patch adds the inner Mistral4 model, routes the outer Mistral3 model to it,
unpacks Mistral4 expert weights, and fixes Pixtral processing with Mistral
tokenizer v15 and later.

Use the software versions required by the vLLM-Ascend 0.26 branch. Do not reuse
the older delivery's CANN or torch-npu pins. The current branch requires CANN
9.1.0, PyTorch 2.10.0, torch-npu 2.10.0.post4, and triton-ascend 3.2.2.

## SHA-256

- vLLM: `c4d5ab1addacebb752e48d001c86e2d0ce636e9736f3969222712d9423ed0a85`
- vLLM-Ascend: `bb2ca5424b76196f189d464acb78fecb866b8a74178a0ea7f1a0420eba5e74a9`

The validated model path uses the Hugging Face-format weights. Remove
`consolidated-*.safetensors` from a mixed-format model directory so vLLM does
not select the native Mistral weight path. Expert parallelism remains disabled
unless the user explicitly enables it.
