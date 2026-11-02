# DSpark APC + draft 窗口外 KV 回收：四组实机验证结果

2026-11-02，910B3 后四卡 TP4。验证在约11:34（Asia/Taipei）完成，早于12:24硬截止。实验使用 `030_dspark_ctxwin_tmp` / `47cff9175f50f47b78c9bf479f8c22b72bf6964e`；vLLM固定 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`。全部请求固定 **temperature=0、top_p=1、seed=42**。

**结论：当前配置下，APC与draft窗口外KV回收可以同时启动和工作，单并发重复、并发4共享长前缀、错峰共享回收及所选块边界均完成。未发现这组取值题因回收而答错；不能据此保证通用精度或逐token一致。**

## 固定运行配置

- target：`/opt/models/Qwen3.8-27B`。
- draft：`export_A_agent_ctxwin2048_step50` 的已有窗口增训权重，使用
  `/workspace/dspark_ctxwin_validation/apc_20261031_round01/models/trained-window-2048` 配置副本。
- 容器：`wyd_va_030rc2_1028`；物理卡4、5、6、7；TP=4。
- MRV2、BF16、async scheduling、FA4 sink=2、7个speculative token、hybrid KV manager开启。
- context_window=2048，context_window_align=128，调度保留量2191。
- max_model_len=32768，max_num_seqs=8，max_num_batched_tokens=8192，gpu_memory_utilization=0.9。
- 请求FULL图模式，实际target为FULL_DECODE_ONLY、draft为FULL；每组独立编译缓存。
- 固定token IDs；输出logprobs=5，未启用会改变APC行为的prompt logprobs。

每组21个请求：10035-token单请求冷/重复/重复；约10028-token取值题并发4、再重复并发4；
10014-token相同前缀错峰并发4；9855/9856/9857-token边界各重复两次。
单请求、取值题和边界样例最多输出128 token，错峰请求最多512 token。
错峰请求在第一条输出约96、192、288 token时加入。
完整启动命令、固定输入、导入路径、源码hash和依赖版本保存在证据归档。

## 四组结果

| 组 | APC | drop | 成功请求 | 实际命中token | 回收块次 | 回收后仍有引用的块次 | 平均接受长度 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | false | false | 21/21 | 0 | 0 | 0 | 2.755 |
| B | false | true | 21/21 | 0 | 172 | 0 | 2.695 |
| C | true | false | 21/21 | 145152 | 0 | 0 | 2.709 |
| D | true | true | 21/21 | 145152 | 55 | 3 | 2.695 |

84/84请求成功，32/32取值回答精确正确（只有4种不同问题，重复运行）；各组抢占均为0。输出token总数依次为2878、2877、2875、2878。平均接受长度定义为 `1 + accepted_tokens / drafts`，仅为运行记录，不作为性能比较。

输入约1万token，明显超过2048窗口。取值答案位于窗口之外；四组均正确，说明本轮target仍能使用长前缀内容。每组4条错峰请求均正常输出到512token上限；这是预设截断，不是模型输出完整答案的证据。检查未发现异常重复循环，最大连续相同token为2。

## APC真实命中和联合边界

C/D完全相同：单请求冷/重复/重复命中 **0 / 9856 / 9856**；共享前缀冷并发4合计29568，热并发4各9856、合计39424；错峰4合计29568；三个边界样例合计26880。全组metrics命中增量 **145152**。

D共有22次lookup、21个请求，其中同一 `cold4_1` 请求被查询两次，两个结果均为9856。原始lookup事件求和155008并不等于真正复用的token数。按request_id去重后为145152，与metrics一致。`results_summary.json`分开保留 `lookup_event_hit_tokens` 和 `request_hit_tokens`，最终数值见 [可读结果摘要](dspark_apc_drop_20261102_evidence/final_analysis.json)。

9855/9856/9857 token输入的热重复分别命中8960/8064/9856，C/D一致。9856样例还记录shared_prefix_boundary=8960，实际共同命中退到8064；没有强行使用某个组单独较长的命中。三个样例在四组内及C/D间输出均逐token相同。

## 回收和多并发的直接证据

D使用 `AscendHybridKVCacheCoordinator`，draft的组ID为2（从0计数） `SlidingWindowManager`，block_size=896、retention=2191；实际读取窗口仍是2048，align=128。16次正命中查询（含一次重复查询）所需连续3个draft尾块全部非null。55个回收块次全部满足已处理token计算的回收边界，引用计数每次恰减1。

错峰请求共享的draft物理块104、逻辑列8、块末8064，记录如下：

| 请求 | processed tokens | 回收边界 | 引用计数 |
| --- | --- | --- | --- |
| D_stagger_0 | 10254 | 8064 | 3 → 2 |
| D_stagger_1 | 10256 | 8066 | 3 → 2 |
| D_stagger_2 | 10255 | 8065 | 2 → 1 |
| D_stagger_3 | 10254 | 8064 | 1 → 0 |

第二次又从3开始，是期间另一个请求加入共享。前三次释放本请求引用后仍保留其他引用；四个请求均完成。这覆盖了实际共享块逐请求退出窗口的情况，不能仅用“target/draft分组独立”来解释正确性。

B观察到153次已回收块重新分配；D未观察到已回收块重新分配，因此**APC开启时缓存驱逐、哈希失效与物理块重用组合尚未实机覆盖**。本轮没有为了触发该路径扩大压力测试。

## 输出差异与精度边界

| 对比 | 完全相同输出 |
| --- | --- |
| A/B：APC关闭，切换drop | 14/21 |
| C/D：APC开启，切换drop | 15/21 |
| A/C：drop关闭，切换APC | 15/21 |
| B/D：drop开启，切换APC | 16/21 |

C/D不同的6条为2条单请求和4条开放式长输出；8条取值题、6条边界请求以及另1条单请求完全相同。四组单请求三次输出长度分别为A=116/116/118、B=115/118/116、C=115/116/116、D=116/118/116。

greedy减少了采样随机性，但本轮仍未实现逐token确定性；具体数值分歧的根因尚未完全定位。APC关闭时自重复已经会分歧。C/D首个分歧处至少一侧最高两个候选并列，另一侧差距0至0.25 log-prob；A/B的相应差距0至0.125。此现象与接近并列候选对数值变化敏感相符，**只是解释性证据，不是证明所有差异都来自浮点误差**。未进行逐步固定共同历史的完整logits对照，也没有通用精度数据集或target-only基准，不能宣布“完全无精度影响”。

另外drop关闭时target/draft FullAttention层合并为一组，总4组；drop开启时draft独立SWA组，总9组。开关同时改变了分组布局。APC开启也使Mamba进入align缓存模式。这些都是当前实现的实际行为，不能把跨组对比解释为只改变了某个物理块的释放动作。

## 代码、环境和异常

实机实验没有改动生产源码；本次整理只提交报告和证据。使用现有解除APC禁令及 `dspark_drop_outside_window_kv` 开关提交，末尾核对关键源码hash未变，本地tmp工作区干净。

新增内容仅为隔离实验脚本、CPU侧缓存审计、固定输入和证据。没有关闭窗口、投机解码或联合命中，没有忽略draft组，也没有重写draft APC策略。

首次A启动因隔离archive缺少已编译自定义算子文件失败；补齐与原服务相同的二进制/vendor软链接，预检通过后重跑。失败完整日志已保留，属于环境装配问题。B正常测试结束后的主动关停触发一次output_handler EngineDeadError，单独记录。A/B/C/D推理期间无异常栈或请求失败，D无异常栈。

NPU4–7在实验前后均显示Alarm，未重置硬件或判定告警无害；本结论是本轮服务/请求通过，不是硬件健康认证。

## 留驻服务与产物

以下仅为实验结束时的状态快照；本次文档整理没有重新连接NPU机器或验证当前服务健康。

实验结束时保留D服务，容器内API PID **609203**，NPU worker PID **609745–609748**；物理卡4–7，`http://127.0.0.1:8874`，模型名 `dspark-apc-experiment`。最终health=200，running=waiting=0。原8873服务已由本轮服务替代，前四卡任务未终止。服务按要求前台运行并tee，不使用nohup。

```bash
ssh 910b3 'docker exec wyd_va_030rc2_1028 curl -fsS http://127.0.0.1:8874/health'
```

- 远端根目录：`/workspace/dspark_ctxwin_validation/apc_drop_20261102_round01`
- 日志：`/opt/w00958190/log/dspark_apc_drop_20261102_round01_{A,B,C,D}_serve.log` 和对应 `_client.log`
- 最终服务快照：`evidence/final_service.json`、`evidence/final_idle.metrics.txt`、`evidence/npu_after.txt`
- 汇总：`final_analysis.json`、`results_summary.json`
- 请求、输出token、logprobs、metrics、缓存事件：`evidence/`
- 分支内完整附件及复核方法：[证据索引](dspark_apc_drop_20261102_evidence/README.md)。
- 上述相对路径均指归档内部；原始复现配置见归档内 `README.md`。

未验证：APC下物理块被驱逐后重用、内存压力/抢占、长时间压力、并发超过4、其他窗口与模型、PD、DCP/PCP、广泛精度基准、设备端所有KV内容逐元素一致性。CPU审计可能影响调度时序，本轮不作性能评测结论。
