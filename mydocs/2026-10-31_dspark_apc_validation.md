# DSpark context window + KV 回收：解除 APC 启动限制的小范围验证

实机验证日期：2026-10-31。补丁与记录整理日期：2026-11-02。

后续APC与回收开关的四组实机对照见 [2026-11-02验证报告](2026-11-02_dspark_apc_drop_validation.md)。
本页继续保留2026-10-31实验结果与当时验证边界。

## 结论与代码范围

仅解除 `validate_windowed_retention()` 对 `enable_prefix_caching=True` 的启动禁令后，
MRV2 DSpark 的 context window、KV 回收和 APC 在本轮小范围实机请求中可以同时工作。
APC 关闭与开启各完成 7 条请求；开启组观察到 6 次真实前缀命中，共 59136 token。
这是指定模型、提交和配置下的功能验证，不是完整 APC 支持、严格数值等价或生产稳定性验收。

实现仅删除 `vllm_ascend/worker/v2/spec_decode/dspark/cache_retention.py` 中的 APC `ValueError`。
不改变草稿窗口读取、回收、KV 分组、hybrid coordinator 联合命中或尾块处理策略，
没有忽略 draft 缓存组，也没有关闭投机解码。
PD connector、DCP/PCP 大于 1、关闭 hybrid manager 的拒绝检查继续保留。
配置测试移除 APC 属于不支持组合的断言，并分别验证 APC 开关下 draft 仍得到相同保留窗口的 `SlidingWindowSpec`。
这项 UT 不验证完整 APC 调度、设备 KV 内容或缓存命中正确性。

## 运行身份与配置

实验基于 Ascend 提交 `ed48548e885aa37126e497c102124184385c636a` 加本次补丁，
vLLM 为 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`。
远端原目录的 791 个源码及相关测试文件与基准内容一致；最终审计确认隔离副本只改了上述实现及测试两个文件。
本次整理前分支为 `93f4158bb799649eae1a413611a78e6474af4eaf`，它相对实验基准只增加了历史报告与附件。

- 主机：`910b3`；容器：`wyd_va_030rc2_1028`；物理 NPU 4、5、6、7；TP=4。
- target：`/opt/models/Qwen3.8-27B`。
- draft：`/opt/w00958190/specforge_tmp/export_A_agent_ctxwin2048_step50` 的真实权重。
  实验目录单独复制 `config.json`，权重软链到原始导出，没有修改原模型配置或重新下载权重。
- MRV2，BF16，seed=42，async scheduling，FA4 sink=2，7 个 speculative token。
- context_window=2048，context_window_align=128，调度保留量=2191。
- max_model_len=32768，max_num_seqs=8，max_num_batched_tokens=8192，gpu_memory_utilization=0.9。
- hybrid manager 开启，组宽参数为 8；两组块池均为 5616 块。
- 请求 FULL 图模式，实际 target 为 FULL_DECODE_ONLY、draft 为 FULL。

实际分组为 FullAttention 8 层两组、draft SlidingWindow 5 层一组、GDN/Mamba 8 层六组，共 9 组。
attention 调度块为 896 token，kernel 块为 128 token。
开启 APC 后，上游自动将 Mamba cache mode 设为 `align`，其调度块由关闭组的 32768 变为 896。
这是相同启动配置切换 APC 后的既有行为，不是额外修改分组实现。

独立 venv 通过 `.pth` 导入实验源码；两组分别使用独立编译缓存和日志。
初始化审计记录导入路径、图捕获及 KV 分组；CPU 侧观察包装器原样转发 lookup 和回收调用。
本轮未启用逐步设备采样，计时含冷编译，不用于性能收益比较。

## 请求、命中与输出

使用固定 token IDs、temperature=0、top_p=1、seed=42，最多输出 128 token。
单并发输入为 10035 token，先冷请求再完整重复两次；随后同时提交 4 条共享长前缀、不同后缀的请求。
五种输入共同前缀为 10011 token，并发输入长度为 10032、10029、10029、10033。

| 项目 | APC 关闭 | APC 开启 |
| --- | --- | --- |
| 服务启动 | 成功 | 成功 |
| 单并发请求 | 3/3 完成 | 3/3 完成 |
| 单并发命中 token | 0、0、0 | 0、9856、9856 |
| 单并发输出 token | 118、116、115 | 118、118、118，逐 token 完全相同 |
| 并发 4 请求 | 4/4 完成 | 4/4 完成 |
| 并发每条命中 token | 均为 0 | 均为 9856 |
| 并发输出 token（按输入 ID） | 68、36、33、101 | 68、36、31、101 |
| 窗口回收的块次 | 56 | 8 |
| 抢占 | 0 | 0 |

两组总计 14 次推理均正常 `stop` 结束，无长度截断；日志未发现服务异常栈，人工检查未见异常循环输出。
APC 开启组的 `prefix_cache_hits_total` 增量为 59136，`prefix_cache_queries_total` 增量为 70228。
调度器 lookup 逐请求返回的命中长度累计同为 59136，证明命中不只是 HTTP 200 或启动成功。
六个命中请求均返回全部 9 组的缓存表；draft 表历史前缀为 null，最后三个物理块非 null。
这确认实际参与了 draft 组联合命中，不能据此声称所有设备读取与写入时序已逐步验证。
回收数是 CPU 管理器观察到的块次，不是不同块 ID 数或完整设备读写证明。

平均接受长度按 `1 + accepted_tokens / draft_steps`，取请求阶段 metrics 差值：
关闭组单并发为 3.5455、4.0345、3.8667，开启组均为 3.7742；并发阶段两组均为 3.0769。
接受长度接近不证明输出等价。

APC 开启组三次输出与关闭组冷请求完全一致，但关闭组自重复从第 88 个输出 token 开始分叉。
并发跨开关对照仅 3/4 逐 token 一致；另一条在第 27 个 token 分叉，输出为 33 与 31 token，措辞不同而内容正常。
未针对该并发后缀追加多轮关闭 APC 的自重复，分叉根因尚未定位。
因此不能把所有差异归因于 APC，也不能把它们全部归为已证明的原有数值波动。

## 验证边界与设备状态

本轮相关完整 UT 为 `tests/ut/worker/v2/test_dspark_context_window_retention.py`，容器内 39 项通过。
此前 40 项独立窗口检查与 6 项提取的配置检查不计作本轮 APC 设备证明；39 项 UT 也不代表全仓测试通过。
本次整理的检查结果另见附件 `integration_checks.json`，不与历史 NPU 结果混算。
2026-11-02 补齐本地格式工具后，全仓 `bash format.sh ci` 的 18 个 manual hooks 通过；
仅添加原始日志短语及机器请求 ID 的精确拼写例外，未改写实机证据。
本地全仓 UT 仍未完成，不能用格式检查替代功能测试。

启动仍有“未识别 draft 专属 KV 组”的上游警告。本轮没有改变该识别或尾块策略。
原始汇总中的“保守回退至所有组尾块保护”属于对源码分支的解释，未直接记录运行时是否走到该分支，
不把它作为尾块保护已独立验证的证据。

实验前后物理卡 4–7 均有 Alarm，最终错误码 `80C98001`；本轮未复位设备。
没有因此判定硬件永久损坏，也不能将本次结果作为告警清除后的设备稳定性验收。

未验证长时间压力、性能收益、其他长度与窗口边界矩阵、高并发、抢占、强制驱逐、
跨请求块复用压力、逐步设备 KV 映射、严格隐藏状态数值等价、MLA、PD、DCP/PCP 或独立 draft APC 优化。
没有扩展成大规模评测，也没有重写 draft prefix cache 策略。

## 证据与复核入口

另一台 910B3 四卡机器的命令与判定标准见 [手动复测步骤](dspark_context_window_manual_validation.md)。

分支内附件见 [证据索引](dspark_apc_20261031_evidence/README.md)，包含原始记录归档及 SHA256 索引。
原始输入、逐请求 token IDs、完整 metrics、调度事件、启动审计、测试与服务日志都在归档内。
初次客户端生成输入时的 `BatchEncoding` 序列化错误也保留在日志中；修正后才发送推理请求，不计作服务失败。

远端实验目录：`/workspace/dspark_ctxwin_validation/apc_20261031_round01`。
完整服务日志位于 `/opt/w00958190/log/dspark_apc_20261031_round01_off_serve.log` 和
`/opt/w00958190/log/dspark_apc_20261031_round01_on_serve.log`，分支内归档亦保存副本。
归档中的 `evidence/serve.sh`、`runtime_manifest.json` 与 `inputs.json` 保存完整启动配置和固定输入。
复跑须新建结果目录和标签，并先核对服务归属、请求是否空闲及设备占用，不能直接覆盖原文件或终止其他任务。

2026-10-31 结束时，APC 开启服务以 `vllm serve` 前台运行、日志 tee 到终端，物理卡 4–7 留驻，
容器内 `127.0.0.1:8873/health` 为 200。该健康记录仅表示当时状态；本次文档整理不重新声明当前服务健康。
实验提前结束，随后一次性硬截止自动任务已删除，未重新启动实验。
