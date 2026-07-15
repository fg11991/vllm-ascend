# DSpark 验证指南(dspark-verify 分支)

本文档说明如何使用本分支在 Ascend NPU 上验证 Qwen3 + DSpark 投机推理。

## 分支说明

`dspark-verify` 基于 upstream main(2026-07-09,`e4568adc`)合入了两个开放 PR,零冲突:

| PR | 路径 | 说明 |
|---|---|---|
| [#11602](https://github.com/vllm-project/vllm-ascend/pull/11602) | **MRV2**(`VLLM_USE_V2_MODEL_RUNNER=1`) | DSpark 算法在 vllm 上游(`DSparkSpeculator`),本 PR 只是薄适配层(slot mapping 转 int32、包 Ascend attn metadata)。**验证以它为主** |
| [#11153](https://github.com/vllm-project/vllm-ascend/pull/11153) | V1(默认 `NPUModelRunner`) | vllm-ascend 内自行实现 DSpark(monkey-patch DFlash 模型挂 Markov/confidence head)。仅支持 FULL_DECODE_ONLY,只实现了 Markov head,作对照参考 |

两条路径互相独立,靠 `VLLM_USE_V2_MODEL_RUNNER` 环境变量切换。

相对 main 的全部改动均为纯 Python(不涉及 csrc),因此**不需要重新编译自定义算子**。

## 版本要求

| 组件 | 要求 | 说明 |
|---|---|---|
| CANN | == 9.0.0(含 NNAL/ATB) | nightly 镜像基于 `cann:9.0.0-910b-ubuntu22.04-py3.12`,已满足 |
| torch / torch-npu | == 2.10.0 | 镜像已满足 |
| triton-ascend | == 3.2.1 | 镜像已满足;注意升级 vllm 后不要弄丢(见下) |
| **vllm** | **main @ `1f486d96a17303ce8db8e02be39545b2be338446`** | 即 `.github/vllm-main-verified.commit`。**镜像里默认是 v0.23.0 tag,必须手动升级**——MRV2 的 dspark speculator、`Qwen3DSparkModel` 原生注册都只在 vllm main 上有 |
| vllm-ascend | 本分支 `dspark-verify` | https://github.com/fg11991/vllm-ascend |
| 宿主机驱动/固件 | 支持 CANN 9.0.0 的 HDK | 宿主机 `npu-smi info` 自查 |

> 用 vllm main `1f486d96` 时,`Qwen3DSparkModel` 已在 vllm registry 原生注册,
> **不需要** PR #11153 描述中提到的手改 `registry.py` 步骤。

## 环境搭建

### 1. 拉镜像

注意:nightly 的 tag 是 `nightly-main`(每日构建);`main` 这个 tag 已停更,不要用。

```bash
docker pull quay.io/ascend/vllm-ascend:nightly-main        # A2/910B
# docker pull quay.io/ascend/vllm-ascend:nightly-main-a3   # A3
# openEuler 系统用 nightly-main-openeuler / nightly-main-a3-openeuler
```

### 2. 容器内升级 vllm 到 main @ 1f486d96(必须)

```bash
cd /vllm-workspace/vllm
git fetch --depth 1 https://github.com/vllm-project/vllm.git 1f486d96a17303ce8db8e02be39545b2be338446
git checkout FETCH_HEAD
VLLM_TARGET_DEVICE=empty python3 -m pip install -e . --extra-index https://download.pytorch.org/whl/cpu/
# 重要:重装 vllm 会带回 x86 版 triton,NPU 上必须卸掉(保留镜像里的 triton-ascend)
python3 -m pip uninstall -y triton
```

### 3. 容器内切换 vllm-ascend 到本分支(纯 Python,无需重装)

```bash
cd /vllm-workspace/vllm-ascend
git fetch https://github.com/fg11991/vllm-ascend.git dspark-verify
git checkout FETCH_HEAD
```

### 4. 自检

```bash
python3 -c "
import subprocess
print('vllm commit:', subprocess.getoutput('git -C /vllm-workspace/vllm rev-parse --short HEAD'))  # 应为 1f486d9
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator
print('dspark import OK')
"
```

## 模型权重

- 目标模型:`Qwen/Qwen3-8B`
- 草稿模型(DSpark):[deepseek-ai/dspark_qwen3_8b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7)(ModelScope 同名)

## 运行验证

speculative-config 两条路径相同(method 沿用 `dflash`):

```json
{"num_speculative_tokens": 7, "method": "dflash", "model": "<dspark 草稿模型路径>"}
```

### MRV2 路径(主要验证对象,PR #11602)

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve Qwen/Qwen3-8B \
  --speculative-config '{"num_speculative_tokens": 7, "method": "dflash", "model": "/path/to/dspark_qwen3_8b_block7"}' \
  --enforce-eager
```

PR 作者的接受率测试用的是 `test_dspark_acceptance.py` 脚本(不在本仓库,在 vllm 上游 examples 或作者本地),参数供参考:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 python examples/test_dspark_acceptance.py \
  --model-dir <Qwen3-8B 路径> \
  --draft-model <dspark_qwen3_8b_block7 路径> \
  --num-spec-tokens 7 \
  --dataset-name hf --dataset-path openai/gsm8k \
  --num-prompts 400 --temp 0 --output-len 256 --enforce-eager
```

也可以直接跑分支自带的 e2e 测试:

```bash
cd /vllm-workspace/vllm-ascend
pytest -sv tests/e2e/pull_request/one_card/model_runner_v2/test_basic.py -k dspark
```

### V1 路径(对照,PR #11153)

不设 `VLLM_USE_V2_MODEL_RUNNER`,其余相同。注意仅支持 FULL_DECODE_ONLY 模式、只有 Markov head。

## 预期结果参考(来自 PR 数据)

| 指标 | MRV2(#11602) | V1(#11153) |
|---|---|---|
| mean acceptance length | 6.48(与 vllm GPU 对齐) | 6.30 |
| 逐位置接受率(token 0) | ~0.95 | ~0.94 |
| 吞吐 | 1270 tok/s(GPU 为 6614,**NPU 吞吐差距是已知现状**) | requests/s 6.62 |

测试条件:gsm8k、temperature=0、num_speculative_tokens=7、enforce-eager。
如果接受长度在 6.3~6.5 附近即符合预期;吞吐低于 GPU 不是环境问题。

## 常见问题

- **启动即报 `ModuleNotFoundError: vllm.v1.spec_decode.llm_base_proposer` 或 `qwen3_dflash`**
  → vllm 没升级到 main `1f486d96`。PR #11153 的 platform patch 在**每次启动时无条件导入**这两个模块,vllm 版本不对时连非 dspark 场景都起不来。
- **报 triton 相关错误**
  → 大概率升级 vllm 时被装回了 x86 triton,`pip uninstall -y triton` 后确认 `pip show triton-ascend` 为 3.2.1。
- **`use_dspark` AttributeError**
  → 同样是 vllm 版本问题,`SpeculativeConfig.use_dspark()` 只在 vllm main 上有。
