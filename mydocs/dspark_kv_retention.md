日期：2026-10-30

# DSpark 窗口外 KV 回收：实现、分组成本与使用边界

本文接续[MRv2 只读窗口报告](dspark_read_window.md)。
设备测量和复现方法集中在[实机验证报告](dspark_npu_validation.md)。
这里解释为什么需要第二阶段、代码怎样实现，以及为什么回收不一定立刻改善启动并发数字。

当前设备验收范围为文本输入的GQA；普通MLA尚未完成设备验收，混合模态未经验证。
回收逻辑会在启用prefix caching、PD connector、DCP/PCP任一大于1，
或关闭hybrid manager时主动拒绝启动。“文本输入”是本次验收范围，不是这里新增的启动检查。

## 从“读不到”到“可以归还”

第一阶段的 draft 只读对齐窗口，但 KV spec 仍是 FullAttentionSpec。
调度器不知道旧历史已经不再需要，因此仍然保留这些块。

原先也一直需要申请池块：“领取”只是把预分配池里的空闲块分配给某个请求，
不是每次生成token都向设备驱动重新申请一块显存。
请求跨过调度块边界时需要更多块；请求结束后归还这些块。

原先16层 target full attention 与5层 draft 可以合为一个21层组。
一个块编号背后有21层各自的 KV 空间；共用编号不等于共用 KV 内容。
target 还要读取旧历史，这个编号就不能归还。
draft 对应的5层空间也不能单独从已分配块中拆出，交给其他请求。

第二阶段把 draft 的保留策略独立出来。
target 继续保留完整历史，draft 的旧块则可以单独返回空闲池。
这里的“释放”是解除请求对预分配 KV 池块的占用，不是要求设备驱动归还整个池。
所以 npu-smi 的进程显存通常不随窗口移动下降。

## 实现为什么放在 get_kv_cache_spec

新增 cache_retention.py，在 draft 加载完成后，找到它声明的 paged KV 层，
仅覆盖这些实例的 get_kv_cache_spec。原方法仍负责产生存储参数：

| 原spec | 启用窗口回收后的spec |
| --- | --- |
| 普通GQA FullAttentionSpec | SlidingWindowSpec |
| 普通MLA spec | AscendDSparkSlidingWindowMLASpec |
| 已有SWA专用spec | 保留原类型，必要时扩大安全保留量 |
| 非attention或None | 保持原样 |

调度器因此使用 SlidingWindowManager 的现有分配和回收路径。
没有另写一套释放器，也没有改变 target 的保留策略。

本实现不设置 Attention/impl 的 sliding_window 属性：
FA4 的选择逻辑会据此判断是否支持该层；FIA也可能改成逐token窗口mask。
这两个效果都不符合本任务。KV spec中的窗口只控制“存多久”；
attention 仍通过第一阶段的对齐目录控制“读哪些”。
这里保留的是第一阶段已经引入的对齐窗口计算语义，不是最初全历史attention语义。
此前基线与回收版服务日志均确认5层draft实际选择FA4；这项结论来自日志，而不只来自属性未改。

W和align都未配置时，回收逻辑不启用；半配置沿用原有报错。
没有新增生产环境变量。

## 为什么保留4239，而不是直接保留4096

不能刚滑出4096就释放，因为对齐会让读起点提前，
投机拒绝、最大长度截断和异步未完成步骤也要留出安全余量。

B为每次draft并行提出的block长度，S为num_speculative_tokens；当前DSpark路径B=S。
A为这个block的起点，即读取器使用的anchor=seq_len-B。
读取区间是[start,seq_len)，长度最多W+align-1+B；W不是固定的总读取长度。
C为本次允许回收的已完成token计数，L为max_model_len。

核对容器实际vLLM v0.30源码ced6857和Ascend准备kernel，调用关系如下：

1. Scheduler._update_after_schedule乐观推进computed时，同步增加in-flight。
2. KVCacheManager.allocate_slots调用remove_skipped_blocks前，使用
   C=max(0,total_computed-in_flight)；_connector_finished也做同样扣减。
   因此即使排队多个批次，回收计数也不会越过最早未完成步骤的起点。
3. Scheduler.update_from_output扣除返回步骤的在途量，并回滚拒绝的token。
   lookahead只扩展需要分配的位置，不直接推进C。
4. prepare_dflash_inputs.py从当前步骤的query range中选最后一个有效位置。
   设当前步骤首位置为P。即使全拒绝，也令last_valid_pos=P-1，
   所以未截断的anchor，即last_valid_pos+1，不小于P。
   正常已处理token路径满足C≤P，因而未截断时C≤A。
5. kernel写出的seq_len=min(last_valid_pos+1+B,L)，读取器再减去B。
   有效位置不超过L-1，因此长度截断最多使anchor后退B。
   于是已核对路径有C-A≤B。实现再多留S，使用较宽的界限C-A≤B+S，
   而不是把async乐观computed直接当作C。

以上推导适用于本报告限定的路径；未来更改scheduler计数或draft输入准备时需要重新核对。
又因为读起点 start 至少为 A-W-(align-1)，选择：

```text
R = W + align + B + S + 1
sliding_window = R
extra_retained_tokens = 0

C - R + 1 <= A - W - align < start
```

manager只释放边界以前的完整块，因此覆盖start的块仍保留。
本次W=4096、align=128、B=S=7，得到R=4239。
R只是回收安全界限，不会扩大attention的实际读取范围。

CPU测试用真实BlockPool和SlidingWindowManager回收，再调用原读窗口代码，
检查有效读取列不含null块。当前步context写入使用原绝对位置slot mapping，
拒绝后缀被屏蔽；大prefill chunk也先分配当前位置再写入。
被清空的旧列既不应读，也不应成为有效写入目标。

## MLA为什么需要额外适配

只把普通MLA换成一个裸SWA类型，会丢失Ascend的布局和存储属性。
反过来，若同时继承FullAttention/MLA类型，uniformity判断又可能把它并回full组。

新AscendDSparkSlidingWindowMLASpec只继承SWA MLA类型，
保留Ascend的存储能力，并注册到SlidingWindowManager。
attn_utils里的归一化、分配与reshape路径识别它；
分组patch也避免把普通MLA误送入DeepSeek V4专用分组逻辑。

speculator初始化时检查draft最终是否仍是SWA组，并打印每组层名、类型、page和block size。
如果planner把它退回full retention，直接报错，避免“配置看似生效，实际上没有回收”。

普通MLA当前只有CPU证据，不应把GQA设备通过当作MLA验收通过。

## 为什么回收后，启动并发估算还可能下降

当前通用设备池给所有组使用统一大小的块。
物理块字节数取各组“层page之和”的最大值。每组独立领取完整块，
不能把一个活跃块内部的剩余层空间再分给其他请求。

本场景的布局为：

| 版本 | full组 | draft组 | GDN组 | 物理块宽度 |
| --- | --- | --- | --- | --- |
| 只读窗口、全量保留 | 21层，含target16+draft5 | 与full合组 | 3组×16层 | 21层 |
| 回收，最小组宽16 | 1组×16层 | 1组×5层 | 3组×16层 | 16层 |
| 回收，最小组宽8 | 2组×8层 | 1组×5层 | 6组×8层 | 8层 |

宽16时draft有11层空位；宽8时有3层空位。
基线的三个GDN组也各有5层空位，比较时不能遗漏。

本次每层page为917504 bytes，即0.875 MiB；它已经包含GDN page统一时的padding。
一个896-token调度块会在worker展开成128-token kernel pages。

启动容量还要为在途prefill保留空间。本配置async最多两个批次，
每批8192 token，SWA每请求保守预算为：

```text
ceil((4239 - 1 + 2×8192) / 896) + 1 = 25块
```

这是容量/准入上界，不是每个稳定decode请求必占25块。
历史报告在max_model_len=32768时测到：
宽16回收版启动估算并发32.65，低于full-retention的35.07；
max_model_len=131072时则从12.51升至14.33。
这些数字可按同一约38.40 GiB池预算复算：基线2139个21层槽，回收版2808个16层槽；
GDN的MambaSpec在本配置使用1个当前状态块加S=7个投机状态块，即每组8块；
这来自max_memory_usage_bytes/manager的投机状态保留规则，并非为每个历史token存KV。
三个GDN组合计24块。32k full需37块，128k需147块。
因此四个并发分别是2139/(37+24)、2808/(37+25+24)、
2139/(147+24)、2808/(147+25+24)。
原因是独立组的空位和在途预留，在较短上下文中抵消了历史回收收益。
这些值也不是实际吞吐。

## size=8为什么值得验证

现有VLLM_ASCEND_KV_GROUP_MIN_SIZE只提高最小组宽，不强制指定宽度。
本模型默认启发式宽度为5，所以设成1不会得到每层一组。
只新增draft_min_size=5也无效：draft已经是5层组，物理块仍由较大组决定。

上表的宽16、宽8分别通过现有变量VLLM_ASCEND_KV_GROUP_MIN_SIZE=16和=8得到；
它们提高本模型的默认下限5，不需要新增分组代码。
CPU真实planner在同一预算下，宽8的启动容量估算优于宽16和宽5，尚不能据此推断吞吐。
宽5让draft填满块，却把空位转移到保留完整历史的target和GDN上。
完全消除层间空位，在保持现有page和块池规则下，需要宽1、共69组；
这会增加manager遍历、block table和slot mapping工作量。

独立draft设备池能让target/GDN用16层块、draft用5层块，
但需要修改池归属、容量、准入、释放、抢占、worker backing与统计。
它还可能出现一池闲置另一池耗尽。它不是新增一个最小组宽参数就能完成的改动。
更详细的候选和工作量见[分组优化规划](dspark_kv_grouping_plan.md)。

## 如何理解先前“32并发每卡约7GiB”的估算

比较对象是上表第一行“全量保留、21层槽”和第三行“回收、8层槽”。
令每层page P=0.875 MiB，full每请求37块、draft每请求6块、GDN每组8块，32请求时：

- draft有效KV减少：32×5×(37-6)×P/1024 = 4.2383 GiB。
- 旧GDN空位减去新draft空位：32×(3×8×5-6×3)×P/1024 = 2.7891 GiB。
- 合计7.0273 GiB/卡。

6块是窗口4239及块内对齐、在途位置共同作用下的稳定decode近似值，
不是所有时刻的硬上限；实机报告直接统计每请求实际块数。
例如C=32768时，完整释放块数floor((C-R+1)/896)=31；
若包括新位置和lookahead的分配末端仍落在第37个块内，
原37个逻辑块只剩索引31～36这6个实体块。位置和在途分配改变时数量会波动。
不能表述成“只靠draft窗口就省7GiB”，也不是整个进程显存下降7GiB。

这个估算假设所有请求都在约32k稳定decode、draft每请求约6块。
保守启动预算使用25块，得到的节省会小很多。
真实请求还会继续增长、存在在途步骤、分批进入和退出。
最终应以实机报告中的实际块计数为准，不把理论值改名成实测值。

## 支持范围和交付状态

启用回收时明确拒绝prefix caching、PD connector、DCP/PCP，
以及关闭hybrid manager的组合。这些拒绝不表示理论上无法支持，
而是当前没有足够证据证明相应命中、传输与分片逻辑正确。

实现提交为cc746b7ca，位于独立分支030_dspark_ctxwin_tmp。
改动覆盖cache_retention、speculator、MLA spec/registry、attn_utils和窄分组兼容处理。
原读窗口算法未重写，未改attention mask，没有重写通用hybrid分组算法。

相关CPU回归389项通过。此前GQA短请求/16.8k请求的eager与图模式对照通过。
新增长请求并发、实际块占用和宽8设备结果另见实机报告。
当前长篇自由续写未达到逐token一致，32并发平均接受长度也有小幅下降；
基线自身的非确定性尚未定位，因此不能宣称生产正确性验收已完成。
生产实现的架构审查、全仓库测试、真实MLA验收不因本文存在而自动完成。
