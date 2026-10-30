日期：2026-10-30

# DSpark MRv2 窗口 KV 回收验证记录

> 本页保留前期验证/规划时的记录。后续size=8并发实测与当前结论见[实机验证报告](dspark_npu_validation.md)；文中的“尚未测试”描述的是当时状态。

分支：`030_dspark_ctxwin_tmp`，基线：`4a01608da`，实现：`cc746b7ca`。
CPU 与指定 GQA 模型的 NPU 对照通过；128k 启动估算并发提升14.53%，32k下降6.89%。
图模式实际为 target FULL_DECODE_ONLY、DSpark FULL，不能宣称支持 GDN 的全 prefill FULL 图。
未推送、未合并。普通 MLA 只有 CPU 证据。

## 改动

- `dspark/cache_retention.py` 在 draft 加载后，仅覆盖该模型声明的 paged cache 层实例的
  `get_kv_cache_spec`。GQA 返回 `SlidingWindowSpec`，普通 MLA 返回
  `AscendDSparkSlidingWindowMLASpec`。不修改 Attention/impl 的 sliding-window 属性。
- MLA spec 仅继承 SWA，保留 Ascend 存储属性及分配/reshape 路径，注册到 `SlidingWindowManager`。
  避免同时继承 FullAttention：上游 uniformity 检查会把这种多继承 spec 并回 full 组。
- 普通 MLA draft 不再触发 DeepSeek V4 专用分组路径；没有改写 hybrid/group-size 算法。
- `_init_context_window` 支持 uniform 包装的 SWA 组，打印每组类型、层名、block size、每层 page bytes；
  若 planner 把 draft 退回 full retention，启动时报错。读窗口算法和 attention 计算均未改动。
- 无窗口配置时保持旧行为，半配置保留原有报错。启用回收时明确拒绝 DCP/PCP、prefix caching、
  PD connector 和关闭 hybrid manager 的组合。没有新增环境变量。

## 保留量推导与读写边界

核对指定容器实际 vLLM v0.30.0 源码 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`：

1. `Scheduler._update_after_schedule` 同时增加 computed 与 in-flight tokens。
2. `KVCacheManager.allocate_slots -> remove_skipped_blocks` 使用
   `C = max(0, total_computed_tokens - num_in_flight_tokens)`，不会直接用 async 乐观计数释放。
3. `Scheduler.update_from_output` 扣除该步 in-flight 数并回滚 rejected tokens；
   `_connector_finished` 的另一条释放路径也减去 in-flight 数。
4. lookahead 只扩展 `num_tokens_need_slot`，不推进释放边界 C。
5. Ascend 实际 kernel `ops/triton/v2/spec_decode/prepare_dflash_inputs.py` 生成
   `A = max(min(last_valid_pos + 1 + B, max_model_len) - B, 0)`。
   全拒绝时从当前步首位置重启，末尾截断使 anchor 最多后退 B。

S 为 `num_speculative_tokens`。已核对的 processed-token 路径中，C 不超过最早未完成步的起点，
故 `C - A <= B`；实现进一步保守地容许 `C - A <= B + S`。
训练窗口起点满足 `start >= A - W - (align - 1)`。采用：

```text
R = W + align + B + S + 1
sliding_window = R
extra_retained_tokens = 0
C - R + 1 <= A - W - align < start
```

manager 仅释放边界以前的完整 scheduler blocks，覆盖 start 的块仍保留。
CPU 测试用真实 `SlidingWindowManager` 释放后再调用原 `DraftContextWindow.apply`，
验证有效读取列没有 null；有效长度之外的补零列不参与 attention。
包含 kernel block 128、scheduler block 896 的组合。

context KV 写入只来自当前 target 步的 query range，拒绝后缀被屏蔽；大 prefill chunk 的当前
range 先分配再写入。slot mapping 使用原始绝对位置表，读侧表替换不影响写入；
`_dflash_local_slot` 还会把 null ID 映射为 `PAD_SLOT_ID`。

## 分组对照

CPU 使用16 full、48 GDN、5 draft 的几何样例，调用真实 planner/config builder：
minimum=16 时由4组、group_size=21变为5组、group_size=16；
未设 minimum 时由4个 uniform 组变为15组。没有需要大改 hybrid 分组的问题。
CPU 样例 page=131072 bytes，不能当作实模型 page。

NPU 实际 TP4、minimum=16，各组每层 page 都为 **917504 bytes**：

| 版本/组 | spec | 层数及层号 | block_size |
| --- | --- | --- | --- |
| 基线0 | FullAttentionSpec | target 16层：3,7,…,63；draft 5层：64,…,68 | 896 |
| 基线1–3 | MambaSpec | 每组16层：0,4,…,60；1,5,…,61；2,6,…,62 | max_model_len |
| 改后0 | FullAttentionSpec | target 16层：3,7,…,63 | 896 |
| 改后1 | SlidingWindowSpec | draft 5层：64,…,68 | 896 |
| 改后2–4 | MambaSpec | 与基线三个 GDN 组相同 | max_model_len |

target 全名为 `language_model.model.layers.{i}.self_attn.attn` 或 `{i}.linear_attn`；
draft 为 `model.layers.{i}.self_attn.attn`。完整列表均已打印到服务日志。
attention kernel block=128，调度块增至896以匹配 GDN page，GDN page padding约9.54%。
基线 group_size=21，draft group IDs=[0]；改后 group_size=16，draft group IDs=[1]。
两版均日志确认5层 draft 选择 FA4。CPU 另验证 FA4/FIA metadata、MLA normalization/allocation/reshape。

## 三阶段验证

环境：SSH `910b3`，容器 `wyd_va_030rc2_1028`，910B3，固定4–7卡。
用户授权后正常结束后四卡原服务，前四卡未操作。全部服务前台运行，日志 tee 到
`/opt/w00958190/log/`。按用户要求，测试结束保留本任务 vLLM 服务占用 4–7 卡：容器内 `127.0.0.1:8872`，模型名 `dspark-retention`，修改后版本、32k、FULL 配置（实际 FULL_DECODE_ONLY）。

配置：TP4，利用率0.9，max-num-seqs=8，max-num-batched-tokens=8192，async，
MRv2，FA4 sink=2，KV_GROUP_MIN_SIZE=16，关闭 prefix caching。
独立 draft 配置 W=4096、align=128，B=S=7，R=4239。原 checkpoint 无窗口配置且未修改；
这里的 W/align 是验证参数，不宣称是该 checkpoint 的训练参数。

| 阶段 | 结果 | 证据等级 |
| --- | --- | --- |
| CPU | 389 passed；包括原窗口测试、真实块回收、async计数、长chunk写入、开关/拒绝配置、分组、MLA布局、FA4/FIA metadata | 已证明：所列样例；对所有调度轨迹属于源码推导与强证据 |
| NPU功能 | 32k eager、32k图模式，改前/后各三条请求；逐token相同、平均接受长度不变 | 已证明：这些请求；不代表所有输入的普遍等价证明 |
| 容量 | 功能通过后完成32k/128k容量对照；差异可由planner公式精确复算 | 已证明：启动日志/metrics数值；满载真实并发和吞吐未压测 |

输入 token 数为26、16826、24，输出分别121、31、60，共212个 token。
每轮均57次 draft、399个 proposed tokens、158个 accepted tokens，
平均接受长度 `1 + 158 / 57 = 3.77192982456`。
128k容量阶段也运行了同一组三条请求：全部六轮输出 token IDs 相同。
输出正常、无替换乱码字符，所有返回 token logprobs 有限；没有对所有内部张量逐一检查 NaN。

请求 FULL 后，GDN backend 将 target 调整为 FULL_DECODE_ONLY；日志确认捕获4个 target 图档位、
3个 DSpark FULL 图档位，图模式实际请求成功。没有强制改变 GDN attention 能力。
仅验证了16.8k实际请求长度，128k是服务 max-model-len，未发送128k长度请求。

CPU日志：`dspark_ctxwin_tmp_cpu_all.log`（389项）、
`dspark_ctxwin_tmp_cpu_retention.log`（分组专项35项）。
NPU日志：`dspark_retention_{baseline,tmp}_{eager,full}_32768.log`，
`dspark_retention_{baseline,tmp}_eager_131072.log`；
对照：`dspark_retention_{eager,full,128k}_compare.log`；
容量汇总：`dspark_retention_capacity_summary.log`。上述文件均在日志目录。
完整 token IDs、原始 metrics、固定 prompts 和复现脚本位于
`/workspace/dspark_ctxwin_validation/`。

## 容量收益与理论差异

以下为固定0.9利用率、eager下的启动容量；32k图模式容量数字也相同。
“KV cache size”是日志按 max-model-len 换算的容量指标，非单层裸 KV token 数。
`max_num_seqs=8` 未用于满载验证这些大于8的估算并发。

| max_model_len | 版本 | num_gpu_blocks | 日志KV cache size/token | Maximum concurrency |
| --- | --- | --- | --- | --- |
| 32768 | 基线 | 2139 | 1149028 | 35.06557 |
| 32768 | 回收 | 2808 | 1069913 | 32.65116 |
| 131072 | 基线 | 2139 | 1639549 | 12.50877 |
| 131072 | 回收 | 2808 | 1877807 | 14.32653 |

128k估算并发提升 **14.53%**，32k下降 **6.89%**，不能宣称两档均有启动容量收益。
用户给定理想公式 `5/21 * (1 - (4096+128+7)/L)` 给出 paged KV 字节节省约20.74%/23.04%；
它不包含 GDN 状态、组 padding、异步未完成批次和块取整，也不能直接当作并发增幅。

实际源码 `VllmConfig.max_in_flight_tokens` 在本配置为2×8192=16384。
`SlidingWindowSpec.max_admission_blocks_per_request` 因而预留
`ceil((4239-1+16384)/896)+1 = 25` 块，而不是只按约4k窗口计费。
三个 GDN 组各预留 `1 + num_speculative_blocks = 8` 个状态块。

基线池槽大小为21×917504=19267584 bytes，改后为16×917504=14680064 bytes；
因此约38.40 GiB相同池预算可装下2139/2808槽。
每请求槽预算，32k为37+24=61对37+25+24=86；128k为147+24=171对147+25+24=196。
分别计算2139/61、2808/86、2139/171、2808/196，与metrics并发值完全一致。
这解释了32k回退和128k改善；未为追求数字改写分组，也未调小安全保留量。

块回收回到预分配池供后续请求复用，不能期待进程的 npu-smi HBM 随窗口移动立即下降。
真实 manager 回池行为已在CPU测到，NPU证据为实际SWA分组、成功长请求与planner容量；
未额外插桩统计NPU运行中每次释放的块数。

## 限制、复现与检查

prefix caching、PD connector、DCP/PCP当前明确拒绝。普通 MLA 有CPU类型/分组/布局证据，
但无真实MLA checkpoint设备验收；特殊压缩/量化MLA、DSV4组合仍需单独验证。
未测试多请求高并发、抢占重算、混合模态输入或非greedy大规模样本。
容器基线的 causal_conv1d import 警告仍存在，但本次图捕获和请求均成功，未改动该路径。

隔离源码：`/workspace/dspark_ctxwin_baseline`（基线额外只加分组日志）与
`/workspace/dspark_ctxwin_tmp`。隔离运行入口：
`/workspace/dspark_ctxwin_validation/runtime-{baseline,tmp}/bin/vllm`。
复用共享编译库和 `_cann_ops_custom/vendors` 的链接，不修改共享安装。
启动脚本 `run-service.sh baseline|tmp eager|full 32768|131072` 固定4–7卡；
每次启动前仍须核对 npu-smi。模型路径为 `/opt/models/Qwen3.8-27B`，
验证draft副本为 `/workspace/dspark_ctxwin_validation/draft-w4096-a128`。

早期启动失败分别来自卡资源竞争、验证CLI缺少main保护、隔离目录缺少CANN vendor链接；
后两项已在两套验证环境补齐，资源问题按用户授权固定后四卡解决。
没有修改 attention 计算来绕过环境问题。
保留服务重启时复用旧编译缓存出现 AOT 解包错误，改用本任务独立的
VLLM_CACHE_ROOT=/workspace/dspark_ctxwin_validation/keepalive-cache-20261030 重新编译；
没有删除共享缓存。保留服务的请求复验另存 tmp_full_keepalive.json。

修改文件 Ruff、codespell、typos、forbidden-import、Markdown 与 diff空白检查通过。
完整 `bash format.sh ci` 仍未完成：环境安装先遇GitHub HTTP/2下载错误，
重试后 nodeenv 获取版本列表遇 `IncompleteRead`。没有运行全仓库测试套件；
该检查缺口仍须在推送/合并前补齐。
