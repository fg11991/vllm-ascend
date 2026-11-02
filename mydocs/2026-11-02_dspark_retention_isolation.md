# 相同 KV 分组下的窗口外 KV 回收隔离诊断

日期：2026-11-02（Asia/Taipei）。20 次单并发请求全部完成。
本次分支整理仅离线复核已有证据，没有重新运行 NPU 请求或检查当前服务。

## 结论

已完成此前未完成的隔离诊断：同一服务进程、同一权重、同一编译图、固定 9 个 KV group，
两边 APC 开启，context window 均为 2048，仅在实验钩子中切换窗口外块的回收。

本样本没有呈现可稳定重复的“实际回收必然降低接受率”。
但这也不是“回收完全无影响”的证明，不能据此否定用户私有模型/数据集上的约 5% 下降。

全部 16 次热请求按 proposed token 加权：实际释放为 2793/3549 = **78.6982%**，
保留为 2704/3381 = **79.9763%**，表面差值 **-1.2781 个百分点**。
其中输出长度与轨迹不完全相同，不能把这个汇总差值直接当作回收的因果效应。

更严格的同缓存对照核实了各 KV group 的命中块 ID 相同。第一次交叉的两对请求，
完整输出均为相同的 418 个 token，但实际释放相对保留分别为 **+1.4389** 和 **-1.4846** 个百分点。
反向重复后又出现输出变化；差异没有稳定跟随开关。
4 对交叉/反向对照中，3 对的首次接受差异出现在 output_tokens_before=56，
而本请求首次实际回收发生在 processed_tokens - prompt_tokens=227 或 228 时。
因此，不能把这些早期差异解释为“本请求刚刚回收的窗口外 KV 被错误读取”。

不过，这不排除先前请求回收改变后续分配布局的间接影响。
本轮没有采集设备 KV 张量逐元素值或完整 draft logits，不能定位数值/分配/调度差异的具体原因，
也没有得出任务答案精度通过的结论。

## 固定条件和唯一实验干预

- 主机 `910b3`，容器 `wyd_va_030rc2_1028`，仅物理 NPU 4、5、6、7，TP4。
- target：`/opt/models/Qwen3.8-27B`。
- draft：`export_A_agent_ctxwin2048_step50`，使用已有 `trained-window-2048` 配置副本。
- 权重 SHA256：`efe3fbf9a844217df975dfb7405cee318f0d20fb5a217a9e0d68da463b4bb0e0`。
- vLLM 提交：`ced6857afa0ea7b2e3f0846a62e1394e90f15607`；启动前工作区干净。
- Ascend 关键生产文件与上一轮及 tmp 分支对应代码的哈希一致；完整信息见 `manifest.json`。
- APC=true，`dspark_drop_outside_window_kv=true`，MRV2、S=7、BF16、seed=42、async、FA4 sink=2。
- context_window=2048，align=128；draft group 2 为 SlidingWindowManager，retention=2191，block_size=896。
- max_model_len=32768、max_num_seqs=8、max_num_batched_tokens=8192、memory utilization=0.9。
- target FULL_DECODE_ONLY、draft FULL；整个对照期间没有重启或重新捕获图。
- 输入为上一轮第一条 agent prompt：固定 **14,512 token**，同一份 token IDs。
- temperature=0、top_p=1、seed=42、max_tokens=512，并发 1。
- 未使用用户私有 checkpoint 或私有数据集。

实验始终保持生产 speculative config 的 drop=true，以固定 KV spec 和分组。
`apc_audit.py` 按请求读取 `retention_mode.txt`，在空队列时切换：

- `release`：调用原始 `SlidingWindowManager.remove_skipped_blocks`，保留原有置空和引用计数释放行为。
- `hold`：跳过该调用，保留原块表及引用计数；读取窗口、联合命中、target 路径和请求结束时的正常释放不变。

这项干预对比的是窗口块退休的整体行为（置空块表并释放引用 vs 保留），
并未进一步把“块表置空”与“物理块引用释放”拆成两个独立开关。
正式生产源码没有修改。钩子先通过四参数真实调用签名、保留路径、释放路径及零边界的独立 CPU 检查。
没有增加设备 tensor.item() 同步，不把带审计日志的时长用作 TPOT 结论。

## 请求顺序与缓存控制

1. 主矩阵：release_a → hold_a → hold_b → release_b，各 1 冷 + 2 热，共 12 次。
2. 每组独立 cache_salt，组内重复相同；4 次冷请求命中均为 0。
3. 复用 release_a 和 hold_b 已有缓存，各切换开关并切回，共 4 次热请求。
4. 对上述两份缓存反向再比较一次，共 4 次热请求；随后停止测试。

所有 16 次热请求各命中 **14,336 token**，累计 **229,376 token**。
后 8 次请求的命中块 ID 与各自原缓存一致，且两种模式之间一致。
这证明命中位置与缓存对象一致，不替代 KV 内容的逐元素校验。

## 分阶段汇总

接受率 = accepted draft tokens / proposed draft tokens；按 token 加权。

| 阶段 | 请求类型 | 实际释放 | 保留 | 输出差异的影响 |
| --- | --- | --- | --- | --- |
| 主矩阵 | 冷 | 663/861 = 77.0035% | 637/770 = 82.7273% | 输出长度不同 |
| 主矩阵 | 热 | 1374/1750 = 78.5143% | 1324/1645 = 80.4863% | 部分输出长度/轨迹不同 |
| 同缓存交叉 | 热 | 710/896 = 79.2411% | 710/896 = 79.2411% | 两对完整 418 token 输出均一致 |
| 同缓存反向 | 热 | 709/903 = 78.5161% | 670/840 = 79.7619% | 两对输出均有变化 |

## 同缓存成对结果

表中的“首次回收”使用已提交 processed_tokens 减去 prompt_tokens 表示，
不是 async scheduler 的乐观计数；它与输出 token 计数不是同一字段。
两者在 56 与 227/228 之间的差距远大于本配置单轮投机长度。

| 阶段/缓存 | 释放−保留（百分点） | 完整输出一致 | 共同输出前缀 | 首次接受差异的 output_tokens_before | 首次回收的 processed−prompt |
| --- | --- | --- | --- | --- | --- |
| cross/release_a | +1.4389 | 是 | 418 | 56 | 227 |
| cross/hold_b | -1.4846 | 是 | 418 | 376 | 227 |
| reverse/release_a | +0.0504 | 否 | 113 | 56 | 228 |
| reverse/hold_b | -2.5991 | 否 | 330 | 56 | 228 |

同一 release_a 缓存、相同实际释放模式，交叉阶段为 355/448（79.24%），
后续反向阶段为 354/455（77.80%）；同样命中且输出都为 418 token。
因此，上一轮出现的 79.24% 与 77.80% 差距，在本轮固定相同回收模式时也能观察到，
不能单凭这两个数值将差异归因于开关。

## 回收和运行证据

- 只出现 1 次 coordinator 初始化，整个实验 scheduler PID=704405，固定 9 组。
- 20/20 成功，metrics 增量与逐请求 scheduler 计数全部一致，preemption=0。
- 实际释放组共 14 次 draft 窗口外回收事件、36 次块释放；边界及 refcount 断言通过。
- 保留组实际释放为 0，记录 332 次有候选块的回收抑制调用。
- 命中所需的 draft 窗口尾块检查全部通过，没有观察到缺失。
- 未记录到已退休块在实验期间通过 get_new_blocks 被重新分配为新槽；未覆盖高缓存占用下的驱逐/复用压力。
- 所有请求自然结束，均未达到 max_tokens=512 的截断上限。

## 全部逐请求计数

| 组 | 重复 | 模式 | accepted/proposed | 接受率 | 输出 token | 命中 token | 实际释放块 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| release_a | 0 | release | 331/434 | 76.2673% | 392 | 0 | 14 |
| release_a | 1 | release | 355/448 | 79.2411% | 418 | 14336 | 1 |
| release_a | 2 | release | 355/448 | 79.2411% | 418 | 14336 | 1 |
| hold_a | 0 | hold | 355/448 | 79.2411% | 418 | 0 | 0 |
| hold_a | 1 | hold | 355/448 | 79.2411% | 418 | 14336 | 0 |
| hold_a | 2 | hold | 355/448 | 79.2411% | 418 | 14336 | 0 |
| hold_b | 0 | hold | 282/322 | 87.5776% | 329 | 0 | 0 |
| hold_b | 1 | hold | 282/322 | 87.5776% | 329 | 14336 | 0 |
| hold_b | 2 | hold | 332/427 | 77.7518% | 392 | 14336 | 0 |
| release_b | 0 | release | 332/427 | 77.7518% | 392 | 0 | 14 |
| release_b | 1 | release | 332/427 | 77.7518% | 392 | 14336 | 1 |
| release_b | 2 | release | 332/427 | 77.7518% | 392 | 14336 | 1 |
| cross_ra_hold | 3 | hold | 354/455 | 77.8022% | 418 | 14336 | 0 |
| cross_ra_release | 3 | release | 355/448 | 79.2411% | 418 | 14336 | 1 |
| cross_hb_release | 3 | release | 355/448 | 79.2411% | 418 | 14336 | 1 |
| cross_hb_hold | 3 | hold | 356/441 | 80.7256% | 418 | 14336 | 0 |
| reverse_ra_release | 3 | release | 354/455 | 77.8022% | 418 | 14336 | 1 |
| reverse_ra_hold | 3 | hold | 332/427 | 77.7518% | 392 | 14336 | 0 |
| reverse_hb_hold | 3 | hold | 338/413 | 81.8402% | 396 | 14336 | 0 |
| reverse_hb_release | 3 | release | 355/448 | 79.2411% | 418 | 14336 | 1 |

## 边界与收尾快照

以下服务状态仅指实机实验结束时，不代表本次文档提交时仍然健康。

这只有一条历史长 prompt、一个已有 checkpoint、单并发、greedy 的小范围实验。
20 次请求不是 20 道独立题目。没有验证用户私有模型/数据集、任务答案正确率、多并发、
高占用缓存驱逐、长时间稳定性、PD 或 DCP/PCP，也没有证明 KV 数值完全正确。
数值非确定性、内存布局或其他执行状态是待验证解释，不是已定位的根因。

本轮开始前，上轮留驻服务已退出且后四卡空闲；保留了原始退出日志，未终止其他任务。
本轮收尾时 `retention_mode.txt` 已恢复为 `release`，APC 和 drop 仍开启。
API PID=704139，EngineCore=704405，worker PID=704424–704427。
`/health` 200、`/v1/models` 正常，最后一个正常释放长请求实际生成 418 token，队列为空。
服务继续前台运行并 tee 到日志，未停止释放卡。原有 NPU Alarm 未在本轮诊断。
实机实验没有修改原模型 config 或 tmp 分支生产代码；本次提交仅整理报告与证据。

远端目录：`/workspace/dspark_ctxwin_validation/retention_isolation_20261102_round02`。
服务日志：`/opt/w00958190/log/dspark_retention_isolation_20261102_round02_I_serve.log`。
完整汇总：`final_summary.json`；原始请求/输出/metrics/回收日志：`evidence/`。
配置、源码与权重指纹：`manifest.json`；证据包包含脚本、日志快照和 SHA256SUMS，不包含权重或编译缓存。

分支内附件、归档指纹和离线复核方法见 [证据索引](dspark_retention_isolation_20261102_evidence/README.md)。
