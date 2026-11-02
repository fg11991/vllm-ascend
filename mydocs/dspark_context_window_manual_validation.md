# 在另一台 910B3 四卡机器上手动验证 context window 与 APC

适用条件：910B3、4 张可用 NPU、已有 Ascend 容器、Qwen3.8-27B target 和窗口增训 draft。
以下命令在容器内的 Bash 执行，无需 Codex。需要两个终端：A 前台运行服务，B 发送请求。
本轮代码已有小范围实机通过记录，可以开始复测；这不表示任意环境或模型组合均已验收。
已知结果和限制见 [实机报告](2026-10-31_dspark_apc_validation.md)。

## 1. 准备代码、环境和路径

本次提交尚未 push，直接拉远端分支可能拿不到 APC 补丁。
将本次交付的 `vllm-ascend-ctxwin-apc.tar.gz` 复制到新容器的 `/workspace`，解压到一个新目录：

```bash
cd /workspace
mkdir vllm-ascend-ctxwin-apc
tar -xzf vllm-ascend-ctxwin-apc.tar.gz -C vllm-ascend-ctxwin-apc --strip-components=1
export ASCEND_SRC=/workspace/vllm-ascend-ctxwin-apc
export VLLM_SRC=/vllm-workspace/vllm

# 改成这台机器已有权重和 FA4 源码的实际路径。
export TARGET=/opt/models/Qwen3.8-27B
export DRAFT=/实际路径/export_A_agent_ctxwin2048_step50
export FA4_SRC=/实际路径/flash-attention-npu

# 示例为这台机器独占的四卡；先用 npu-smi 核对物理编号与占用。
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
npu-smi info
export RUN_ROOT=/workspace/dspark-manual-$(date +%Y%m%d-%H%M%S)
export LOG_DIR=/opt/w00958190/log
mkdir -p "$RUN_ROOT" "$LOG_DIR"
```

如果容器映射的是物理卡 4–7，应按实际可见编号调整，不照抄 0–3。
只使用自己可用的卡；不要按进程名批量停止服务。

该验证依赖配套 vLLM 开发代码，不是仅安装普通发行版 vLLM 即可复现。
先确认 vLLM 源码提交、环境和模型配置：

```bash
test "$(git -C "$VLLM_SRC" rev-parse HEAD)" = ced6857afa0ea7b2e3f0846a62e1394e90f15607
git -C "$VLLM_SRC" status --short
python - <<'PY'
import importlib.metadata as m
import json
import os
from pathlib import Path

for name in ('torch', 'torch-npu', 'transformers', 'triton', 'vllm', 'requests'):
    print(name, m.version(name))
cfg = json.loads((Path(os.environ['DRAFT']) / 'config.json').read_text())
window = cfg['dflash_config']
assert window['context_window'] == 2048, window
assert window['context_window_align'] == 128, window
assert cfg['block_size'] == 7, cfg['block_size']
print('draft window:', window['context_window'], window['context_window_align'])
assert (Path(os.environ['TARGET']) / 'config.json').is_file()
PY
```

此前验证使用 Python 3.12.13、torch 2.10.0+cpu 配合 torch-npu 2.10.0.post7、
Transformers 5.14.1、Triton 3.6.0，以及 FA4 提交 `64a5652de4ba158083c43a6d4ed0a474a7a11419`。
这里的 torch `+cpu` 是该 torch-npu 环境的包标识，不表示推理使用 CPU。
复用已有兼容容器，不为执行本步骤自动升级 Transformers、CANN 或驱动。
如 vLLM 提交不同或工作区已有修改，先确认差异；不能把不同实现的结果直接并入本轮结论。

在这个专用实验容器中安装刚传来的 Ascend 源码，保留已有依赖版本：

```bash
python -m pip install --no-deps --no-build-isolation -e "$ASCEND_SRC"
cd /workspace
python - <<'PY'
import vllm
import vllm_ascend
print('vllm:', vllm.__file__)
print('ascend:', vllm_ascend.__file__)
PY
```

确认 Ascend 导入路径指向新目录，vLLM 指向上述配套源码。
安装需要容器已有的构建依赖与 CANN 工具链；构建失败应保留日志，不用旧包继续宣称验证了新补丁。
FA4 也需容器已有可用安装或源码及其依赖。不要修改原始权重的 `config.json`。

## 2. 准备相同输入和轻量审计

复用归档中的固定输入，避免不同分词或不同 prompt 干扰对照。
这些 token IDs 只适用于本轮 Qwen3.8-27B 的相同 tokenizer；换模型需重新制作输入。

```bash
mkdir "$RUN_ROOT/archive"
tar -xzf "$ASCEND_SRC/mydocs/dspark_apc_20261031_evidence/raw_evidence.tar.gz" \
  -C "$RUN_ROOT/archive"
for f in inputs.json run_requests.py apc_audit.py startup_audit.py summarize.py; do
  cp "$RUN_ROOT/archive/evidence/$f" "$RUN_ROOT/$f"
done
cat > "$RUN_ROOT/run_vllm.py" <<'PY'
from vllm.entrypoints.cli.main import main
from apc_audit import install

install()

if __name__ == '__main__':
    main()
PY
declare -p ASCEND_SRC VLLM_SRC TARGET DRAFT FA4_SRC ASCEND_RT_VISIBLE_DEVICES RUN_ROOT LOG_DIR \
  > "$RUN_ROOT/env.sh"
echo "终端 B 也需要 source $RUN_ROOT/env.sh"
```

审计只记录初始化和 CPU 侧 lookup/回收事件，不修改缓存策略，不做逐步设备拷贝。
归档客户端固定使用 `127.0.0.1:8873` 和模型名 `dspark-apc-experiment`，以下服务与之保持一致。

## 3. 终端 A：先启动 APC 关闭基线

```bash
cat > "$RUN_ROOT/serve_manual.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
tag=${1:?use off or on}
case "$tag" in
  off) apc_flag=--no-enable-prefix-caching ;;
  on) apc_flag=--enable-prefix-caching ;;
  *) echo 'use off or on' >&2; exit 2 ;;
esac
test ! -e "$RUN_ROOT/$tag.scheduler.jsonl"
test ! -e "$LOG_DIR/$(basename "$RUN_ROOT")_${tag}_serve.log"
export DSPARK_FOLLOWUP_TAG="$tag"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK=2
export VLLM_ASCEND_KV_GROUP_MIN_SIZE=8
export VLLM_CACHE_ROOT="$RUN_ROOT/cache-$tag"
export TRITON_CACHE_DIR="$VLLM_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$VLLM_CACHE_ROOT/inductor"
export TORCH_EXTENSIONS_DIR="$VLLM_CACHE_ROOT/torch_extensions"
export ASCEND_CACHE_PATH="$VLLM_CACHE_ROOT/ascend"
export TORCHAIR_CACHE_HOME="$VLLM_CACHE_ROOT/torchair"
export OMP_NUM_THREADS=1
export PYTHONPATH="$FA4_SRC:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ASCEND_SRC/vllm_ascend/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" \
  "$ASCEND_CACHE_PATH" "$TORCHAIR_CACHE_HOME" "$VLLM_CACHE_ROOT/runtime"
spec=$(python -c 'import json,os; print(json.dumps(dict(method="dspark", num_speculative_tokens=7, model=os.environ["DRAFT"])))')
cd "$VLLM_CACHE_ROOT/runtime"
python "$RUN_ROOT/run_vllm.py" serve "$TARGET" \
  --served-model-name dspark-apc-experiment --host 127.0.0.1 --port 8873 \
  --dtype bfloat16 --seed 42 --max-model-len 32768 --max-num-seqs 8 \
  --max-num-batched-tokens 8192 --tensor-parallel-size 4 --gpu-memory-utilization 0.9 \
  --trust-remote-code "$apc_flag" --async-scheduling \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --speculative-config "$spec" \
  --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[8,16,24,32,40,48,56,64]}' \
  2>&1 | tee "$LOG_DIR/$(basename "$RUN_ROOT")_${tag}_serve.log"
SH
bash "$RUN_ROOT/serve_manual.sh" off
```

服务保持前台运行，不使用 nohup。首次加载与编译需要等待，不用短超时反复重启。
日志应显示 2048/128 读窗口、5 层 draft 的 SlidingWindowSpec、2191 保留量和 9 个 KV 组。
实际图模式应为 target FULL_DECODE_ONLY、draft FULL。
如果无法启动，不通过关闭窗口、回收或投机解码绕过问题。

## 4. 终端 B：发送 7 条请求，再切换 APC

把下面 env.sh 路径改成终端 A 打印的真实路径：

```bash
source /workspace/dspark-manual-实际时间/env.sh
set -o pipefail
curl -f http://127.0.0.1:8873/health
curl -f http://127.0.0.1:8873/v1/models
test ! -e "$RUN_ROOT/off.before.metrics.txt"
python -u "$RUN_ROOT/run_requests.py" off \
  2>&1 | tee "$LOG_DIR/$(basename "$RUN_ROOT")_off_client.log"
curl -fsS http://127.0.0.1:8873/metrics | grep -E '^vllm:num_requests_(running|waiting)\{'
```

客户端依次执行 10035-token 冷请求、两次完整重复，再做共享 10011-token 前缀的并发 4 请求，
temperature=0、最多输出 128 token。等客户端完成且 running/waiting 均为 0，
在终端 A 按 Ctrl-C 停止自己刚启动的基线，等该服务退出后启动：

```bash
bash "$RUN_ROOT/serve_manual.sh" on
```

终端 B 等服务重新就绪后执行：

```bash
curl -f http://127.0.0.1:8873/health
test ! -e "$RUN_ROOT/on.before.metrics.txt"
python -u "$RUN_ROOT/run_requests.py" on \
  2>&1 | tee "$LOG_DIR/$(basename "$RUN_ROOT")_on_client.log"
python "$RUN_ROOT/summarize.py" > "$RUN_ROOT/summary.stdout.json"
```

每组只运行一次；需要重试时使用新的 RUN_ROOT，保留首次错误及旧结果。
命令失败后先查看日志，不带着不完整结果继续下一个阶段。

## 5. 判断通过的依据

查看 `results_summary.json`、`off/on.scheduler.jsonl`、各阶段 metrics 与服务日志。

- 两组各 7 条请求完成，输出 token IDs 长度与 usage 一致，没有异常循环、卡死或服务错误。
- APC 开启组必须有大于 0 的 `lookup.hit_tokens`，并与 `prefix_cache_hits_total` 增量核对。
  上一轮两次重复及 4 条并发请求各命中 9856 token，总计 59136；它是参考结果，不替代本机实测。
- lookup 应包含 draft 组，不能只看到 target 命中；历史前缀可为 null，但有效窗口尾部应有驻留块。
- `window_reclaim` 事件的 retention 为 2191。上一轮关闭组回收 56 块次、开启组 8 块次。
  命中后跳过大部分 prefill，回收计数减少本身不是禁用了回收。
- 记录平均接受长度及输出差异。上一轮开启组单并发输出完全相同，但关闭组自重复和一条并发对照有分叉。
  若本机出现差异，补充同输入的 APC 关闭自重复对照，不直接归因于 APC，也不据此宣称数值等价。

保留 `$RUN_ROOT` 和 `$LOG_DIR` 内本轮日志。前台健康服务可继续留驻，不自动释放卡。
这套小组请求不覆盖生产稳定性、长时间压力、强制驱逐、抢占、更多窗口边界、MLA、PD 或 DCP/PCP。
本机若已有硬件告警应单独记录；之前结果包含 `80C98001` 告警背景，不能当作告警清除后的稳定性证明。
