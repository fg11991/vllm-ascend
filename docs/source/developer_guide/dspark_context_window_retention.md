# DSpark MRv2 窗口 KV 回收验证记录

日期：2026-10-30。分支：`030_dspark_ctxwin_tmp`，基线：`4a01608da`。
实现提交：`cc746b7ca`。尚未完成 NPU 验收，不应据此合并或宣称显存收益已验证。

## 改动

- `dspark/cache_retention.py` 在 draft 加载后，仅覆盖该模型声明的 paged cache 层实例的 `get_kv_cache_spec`。
  GQA 返回 `SlidingWindowSpec`；普通 MLA 返回 `AscendDSparkSlidingWindowMLASpec`。
  不修改 Attention/impl 的 sliding-window 属性，不改变 FA4 选择、FIA mask 或读窗口算法。
- MLA 的新 spec 仅继承 SWA，保留 Ascend 存储属性及分配/reshape 路径；注册到 `SlidingWindowManager`。
  避免同时继承 FullAttention：上游 uniformity 检查会把这种多继承 spec 并回 full 组。
- 普通 MLA draft 不再触发 DeepSeek V4 专用分组路径。没有改写 hybrid 分组算法或 group-size 策略。
- `_init_context_window` 支持 uniform 包装的 SWA 组，打印每组类型、层名、block size、每层 page bytes；
  若 planner 把 draft 退回 full retention，启动时报错。
- 无窗口配置时保持旧行为；原先的半配置报错也保留。启用窗口回收时明确拒绝 DCP/PCP、prefix caching、
  PD connector 和关闭 hybrid manager 的组合。没有新增环境变量。

## 保留量推导与读写边界

核对的是指定容器实际 vLLM v0.30.0 源码 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，不是本机 main：

1. `Scheduler._update_after_schedule` 同时增加 `num_computed_tokens` 与 `num_in_flight_tokens`。
2. `KVCacheManager.allocate_slots` 调用 coordinator 的 `remove_skipped_blocks` 时传入
   `C = max(0, total_computed_tokens - num_in_flight_tokens)`。异步队列的乐观计数因此不直接用于释放。
3. `Scheduler.update_from_output` 扣除该步 in-flight 数并回滚 rejected tokens；
   `_connector_finished` 的另一条释放路径也减去 in-flight 数。
4. lookahead 只扩展 `num_tokens_need_slot`。它不会把释放边界向前推进。
5. Ascend 实际使用 `ops/triton/v2/spec_decode/prepare_dflash_inputs.py`：
   `A = max(min(last_valid_pos + 1 + B, max_model_len) - B, 0)`。
   全拒绝时从当前步第一个位置重启；末尾截断使 anchor 最多后退 B。

令 S 为 `num_speculative_tokens`。在已核对的 processed-token 路径，C 不超过最早未完成步的起点，
故 `C - A <= B`；实现进一步保守地容许 `C - A <= B + S`。
训练窗口起点满足 `start >= A - W - (align - 1)`。采用：

```text
R = W + align + B + S + 1
sliding_window = R
extra_retained_tokens = 0
C - R + 1 <= A - W - align < start
```

manager 只释放边界以前的完整 scheduler blocks；即使 scheduler block 大于 kernel block，
覆盖 start 的 scheduler block 仍保留。CPU 测试用真实 `SlidingWindowManager` 释放块，再运行原
`DraftContextWindow.apply`，验证所有有效读取列非 null；clone 超出有效长度的补零列不参与 attention。

context 写入只来自当前 target 步的 query range，拒绝后缀被屏蔽；大 prefill chunk 的整个当前 range
都先分配再写入。slot mapping 使用原始绝对位置表，`_dflash_local_slot` 还会把 null block 映射为
`PAD_SLOT_ID`。读侧表的替换不影响这些写入。

## 分组对比：CPU planner 样例

使用 Qwen 的 16 个 full 层、48 个 GDN 层、5 个 GQA draft 层。样例采用 TP4 的 GQA 几何，
`block_size=128`，每层 page 为 131072 bytes；GDN 使用简化 payload 并 padding 到同一 page。
调用了实际分组函数和 `get_kv_cache_config_from_groups`，但这不是实模型启动后的缓存配置。

| 设置 | 改前 | 改后 |
| --- | --- | --- |
| `KV_GROUP_MIN_SIZE=16` | full×21；GDN×16、16、16；group_size=21，共4组 | full×16；GDN×16、16、16；SWA draft×5；group_size=16，共5组 |
| 未设 minimum | Kimi 形状匹配：full uniform×21，GDN uniform×16、16、16，共4组 | full 分4组，GDN分10组，SWA draft单组，共15组 |

target full 层号为 `3,7,...,63`，draft 为 `0,...,4`，其余 target 层为 GDN。
完整层列表、每组 spec/block/page 已打印在 `dspark_ctxwin_tmp_cpu_retention.log`。
uniform spec 的 page bytes 是组内 page 总和，不能直接当成每层 page 比较。
专用 draft SWA 组及 uniform 包装均通过读窗口初始化测试；真实服务的 group IDs 仍待验收。

## 三阶段验证状态

| 阶段 | 结果 | 证据等级 |
| --- | --- | --- |
| CPU | 389 passed：原窗口测试、回收不变式、异步乐观计数样例、长 chunk 写入、开关与拒绝配置、分组、MLA布局、FA4/FIA metadata | 已证明：所列 CPU 样例通过；对完整设备运行属于强证据 |
| NPU 功能 | 基线 eager 启动失败：初始化期间空闲卡被另一组进程占用；未产生推理结果 | 未验证：greedy token一致性、接受长度、FULL/eager长请求、NaN |
| 显存收益 | 按顺序要求未进入该阶段；没有32k/128k实测值 | 推测：窗口外块可回池复用，收益幅度待测 |

CPU 在 `910b3` 的 `wyd_va_030rc2_1028` 容器内运行，隔离源码位于 `/workspace/dspark_ctxwin_tmp`。
日志为 `/opt/w00958190/log/dspark_ctxwin_tmp_cpu_all.log` 与 `dspark_ctxwin_tmp_cpu_retention.log`，均 tee 到终端。
后者是分组专项记录（35项），前者是最终相关测试集（389项）。没有运行全仓库测试套件。
修改文件 Ruff、codespell、typos、forbidden-import 检查、报告 Markdown 检查与 `git diff --check` 通过。
完整 `bash format.sh ci` 未通过环境初始化：首次 GitHub HTTP/2 下载失败，改用 HTTP/1.1 后
在 markdownlint 的 nodeenv 下载版本列表时发生 `IncompleteRead`，尚不能宣称全量格式检查通过。

NPU 在两次 `npu-smi` 确认0–3卡无进程、各约3.4 GiB基础占用后启动，4–7卡保持其他任务使用。
基线初始化期间，0–3卡出现另一组进程（设备侧 PID 264376、264377、264386、264387），
占用升到约55 GiB；基线报 `Free memory ... (9.64/60.96 GiB) ... less than ... (0.9, 54.86 GiB)` 后退出。
日志：`/opt/w00958190/log/dspark_retention_baseline_eager_32768.log`。
没有终止其他任务，没有降低利用率挤占其他服务；本任务待发请求客户端已停止。
更早一次启动暴露的隔离 CLI 缺少 `__main__` 保护已修正，仅涉及验证入口。

## 收益预期与风险

用户给出的理论值衡量理想的 paged KV 字节节省。实测还须计入 group padding、scheduler block 向上取整、
GDN 状态块以及 `SlidingWindowSpec.max_memory_usage_bytes` 的 `max_in_flight_tokens` 预算。
本样例中5层 draft 的 SWA 组按16层组宽复用池，不能直接把所有收益都按5层计算。
短 max-model-len 或较大的 prefill/in-flight 预算可能降低甚至抵消启动日志中的 concurrency 改善。
`num_gpu_blocks` 也随组宽变化，必须结合 page/group bytes 对比；回收的是预分配池内的块，
不等价于 `npu-smi` 的已分配 HBM 立即下降。当前没有证据判断32k/128k是否符合预期。

prefix caching 与 PD 已明确拒绝，不能宣称支持。普通 MLA 有类型、manager、分组隔离、FIA/FlashMLA
存储布局的 CPU 证据，但无真实 MLA checkpoint NPU 验收。特殊压缩/量化 MLA、DSV4 混合组合仍需单独验证。
容器导入原基线还会提示 `causal_conv1d_update_npu` 缺失并回退；此问题不在本次修改范围，
FULL graph 验收时须区分它与 retention 回归。

## 继续验收的位置

target：`/opt/models/Qwen3.8-27B`；原 draft：`/opt/models/Qwen3.8-27B-DSpark-Base`。
原 draft 未配置窗口，验证副本 `/workspace/dspark_ctxwin_validation/draft-w4096-a128`
仅复制 config 并链接权重，设置 W=4096、align=128；共享 checkpoint 未修改。

基线代码：`/workspace/dspark_ctxwin_baseline`（4a01608da）；新代码：`/workspace/dspark_ctxwin_tmp`。
各自的隔离 Python/CLI 位于 `/workspace/dspark_ctxwin_validation/runtime-{baseline,tmp}/bin/`，
复用现有依赖与编译库，不修改当前已安装包或其他服务的环境。
基线隔离副本仅额外加入分组诊断日志，不改变计算或保留行为。
前台启动脚本为 `/workspace/dspark_ctxwin_validation/run-service.sh`，参数依次为
`baseline|tmp`、`eager|full`、max-model-len；请求脚本为同目录下 `validate_outputs.py`。
脚本目前指定0–3卡，重跑前必须核对并改为届时实际空闲卡。FULL模式尚未尝试。

先在容器内确认连续空闲卡且没有正在加载的服务，再从 `/workspace` 前台运行；不使用 nohup。
两版本均使用同一验证副本、`VLLM_USE_V2_MODEL_RUNNER=1`、`VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK=2`、
`VLLM_ASCEND_KV_GROUP_MIN_SIZE=16`、`--no-enable-prefix-caching`、greedy、相同 TP/利用率/批预算。
先做 eager 与 FULL 的长请求逐 token 比较及接受长度，再做32k、128k容量对照。
`/v1/completions` 的 `return_token_ids=true` 可直接记录 token IDs；接受长度使用专用服务的
spec-decode counters 差分，避免混入其他请求。尚未完成这些步骤，不应把准备好的环境视为验收通过。
