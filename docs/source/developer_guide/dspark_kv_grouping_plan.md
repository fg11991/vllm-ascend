# DSpark KV 分组空位优化规划

> 本页保留前期验证/规划时的记录。后续size=8并发实测与当前结论见[实机验证报告](dspark_npu_validation.md)；文中的“尚未测试”描述的是当时状态。

日期：2026-10-30。范围：Qwen3.8-27B target + 5 层 GQA DSpark，
MRv2、TP4、W=4096、align=128、B=S=7、async、
max_num_batched_tokens=8192，prefix caching/PD/DCP/PCP 仍关闭。
本次仅阅读源码和运行 CPU planner；未改生产实现、未重启服务。

## 结论

可以减少空位，也可以设计完全无层间空位的方案。
但仅增加 draft 专用 group-min-size 参数不能改变当前共享池的物理分配粒度。
建议首先验证现有 VLLM_ASCEND_KV_GROUP_MIN_SIZE=8；
它无需生产代码改动，CPU planner 已证明优于当前16和候选5的容量估算。
这里的证明限于固定预算和指定几何，不能代替设备吞吐、延迟和真实并发验证。

## 源码依据

以下 vLLM 路径均位于容器 /vllm-workspace/vllm，
Ascend 相对路径位于本分支。

| 位置 | 实际约束 |
| --- | --- |
| vllm_ascend/patch/platform/patch_kv_cache_utils.py:99–163 | 分组宽度为 max(启发式默认值, min_size)，本模型默认5；min_size=1并不强制变成1 |
| vllm/v1/core/kv_cache_utils.py:1588–1611 | 物理池块字节数取所有组的每层 page 字节之和的最大值 |
| vllm/v1/core/kv_cache_utils.py:1745–1798 | available_memory / bytes_per_block 得到统一 num_blocks；各组视图从同一 backing 起点覆盖 |
| vllm/v1/core/kv_cache_coordinator.py:98–152 | 一个设备 BlockPool 传给各组 manager |
| vllm/v1/core/kv_cache_coordinator.py:171–230 | 各组所需块数直接相加 |
| vllm/v1/core/kv_cache_manager.py:519–561 | 准入/本步分配与单个池的空闲块数比较 |
| vllm/v1/kv_cache_interface.py:1414–1436 | 通用设备容量只有 num_blocks；HiSparse host 的独立容量不是第二个通用设备池 |
| vllm_ascend/worker/v2/attn_utils.py:1174–1195 | hybrid descriptors 要求一个共同 backing，并在 worker 只分配一次 |
| vllm_ascend/worker/v2/block_table.py:75–110 | slot mapping 存储与 kernel grid 包含 num_groups 维度 |
| vllm_ascend/worker/v2/spec_decode/dspark/context_window.py | 已按 group_ids 循环维护窗口副本；多 draft 组仍需集成验证 |

因此 target/GDN 保持16层组、draft设5层组，仍然得到16层宽的物理块。
draft 本来就只有一个5层组，独立参数设5在这里不产生新效果。
若再把 draft 切成多个更小组，而其他组不变，物理块仍宽16，反而消耗更多块。
不能让不同请求/组直接复用一个活跃块的“空位”，否则需要新的子块所有权和地址映射。
把 GDN 或 full 层直接塞进 draft SWA 组也不成立：它们的状态寻址和回收规则不同。

## 已运行的 CPU planner 对照

调用真实 Ascend grouper、get_kv_cache_config_from_groups 和
get_max_concurrency_for_kv_cache_config；只用 CPU，不加载模型权重。
使用实测几何：每层 page=917504 bytes、attention scheduler block=896、
16 full + 48 GDN + 5 SWA；GDN每组8状态块、SWA每组25块预算。

统一池预算固定为2808×16×917504 bytes，避免候选之间被额外显存预算干扰。
Mamba只构造等 page 的简化状态形状；不是完整模型初始化。
16/8/5使用现有min_size路径；4/2/1由临时脚本显式拆组后交给原planner，
属于方案可表示性试验，不是已实现的启动参数。

| 分组宽度 | 组数 | draft占用/空位 | 32k估算并发 | 128k估算并发 |
| --- | --- | --- | --- | --- |
| 16，现状 | 5 | 5层使用、11层空位 | 32.6512 | 14.3265 |
| 8，现有参数可用 | 9 | 5层使用、3层空位 | 38.2041 | 15.3025 |
| 5，现有参数可用 | 15 | draft无空位，target/GDN出现空位 | 35.5138 | 12.9654 |
| 4，显式拆组原型 | 18 | draft两组3/2层，共3层空位 | 38.2041 | 15.3025 |
| 2，显式拆组原型 | 35 | draft三组2/2/1层，共1层空位 | 39.9005 | 15.5676 |
| 1，显式拆组原型 | 69 | 所有组无层间空位 | 40.8065 | 15.7036 |

宽8：target拆2组、GDN拆6组、draft仍1组。
池块数5616，32k每请求预算2×37+6×8+25=147，128k为2×147+6×8+25=367。
故得到5616/147与5616/367。相对当前16，分别提高17.01%与6.81%的估算容量。
相对此前full-retention设备日志35.0656/12.5088，约提高8.95%/22.33%；
后者只是跨方案估算对照，尚无同参数NPU性能结论。

宽5并不减少整个系统的padding：16层拆4组，每组4层，合计4个空位；
48层拆10组，其中两个4层组，合计2个空位。
这些空位对应 full 长历史和 GDN 状态，成本不能只按空位个数比较。
宽4与宽8的容量估算相同，但多一倍组数，所以宽4在本次静态结果中没有优势。

在固定每层page、固定tokens/block、类型独立且仍共享等大块池的前提下，
所有组完全无层间空位要求宽度同时整除16、48、5，最大公约数为1，因而需要69组。
这里不包括GDN page padding、token块边界空位，以及保守在途预算；
“无层间空位”不等于没有任何内存开销。

## 方案与工作量

以下为单人熟悉代码情况下的粗估人日，包括指定范围验证，不是交付承诺。
实际耗时取决于NPU编译、性能回归和多组实现中的隐含假设。

### A. 先验证宽8：推荐第一步

- 仅修改独立验证启动脚本的现有参数，不新增环境变量。
- 固定W/align/R、batch token预算、模型长度和利用率。
- CPU确认9组、每层page相等、draft group ID和布局不变。
- NPU对照16与8：eager/图模式，greedy逐token、接受长度、长请求。
- 增加多请求测试，测TTFT、TPOT、输出吞吐、抢占次数；不要继续仅用max_num_seqs=8验证日志估算并发。
- 分别做固定请求数的性能对照和长请求容量爬坡，不能用日志并发代替实际测试。
- 预计0.5–1人日；生产逻辑改动0行。每次服务切换后保留可用vLLM服务占后四卡。

### B. 窄范围分组选择：A证实收益后再考虑

- 只针对启用窗口保留且page几何兼容的DSpark组合，不影响默认其他模型。
- 候选宽度生成真实分组，按物理块字节数×每请求块预算评价。
- 同时约束组数；容量优先不能无条件选69组。
- 明确“最小宽度”和“指定宽度”的不同语义，保留用户显式配置的优先级。
- 不先增加draft专用min-size；若需要可配置策略，先评审接口，集中定义并文档化。
- 预计2–4人日；主要为分组patch、配置接入、分组/容量UT及设备回归。
- 暂不改scheduler allocator、attention、读窗口和保留量。

### C. 完全去掉层间空位，保留现有池：69组实验

- 新增受控的显式宽度/拆组入口；现有min_size=1不会实现该目标。
- draft会分成5组，检查每层到组映射、context KV写入、窗口副本、FA4 metadata、图捕获。
- CPU模拟分配/释放并检查组间不重叠；设备检查输出与多请求延迟。
- manager遍历、block table、slot mapping的工作量随组数增加；真实性能影响未测。
- 窄范围实验预计2–4人日；若发现需要改热路径，另行评估。
- CPU估算比宽8仅再多约6.81%/2.62%容量，应先确认这点收益是否值得额外开销。

### D. draft独立设备池：能保留少量组，但属于架构改动

- target/GDN池用16层宽块，draft池用5层宽块，分别管理空闲块。
- KVCacheConfig须表达pool归属、各池容量和各层视图；
  coordinator须路由manager，准入与抢占必须逐池判断，不能再只比较一个空闲块总数。
- worker须真正分配两个backing，并按池容量reshape；
  不能只缩小draft tensor却继续使用全局block ID，否则可能越界或互相覆盖。
- 两池之间还须分配显存预算：固定比例可能出现一池闲置、另一池耗尽；
  动态调剂需要重映射/迁移，复杂度更高。
- 块事件、usage统计、重算/取消/异常释放、多TP rank一致性也要适配。
- 固定两池、仅本模型且维持prefix/PD/CP拒绝的原型预计5–10人日；
  完整回归与可维护集成预计2–4周。增加动态调剂或prefix/PD支持另计。
- 最优预算下可接近宽1的静态容量上界，但固定分池未必达到；没有设备证据。
- 不建议为一个draft min-size参数把这些改动隐藏进当前小补丁。

## 验收与下一步

优先A，设备收益成立后决定是否需要B。C/D作为后续选项，不自动展开。
本次已证明：指定几何下分组与容量计算结果、单改draft宽度不能缩小全局块。
尚未证明：宽8/5/1的设备性能、真实可承载并发、独立池的实际收益。
未修改生产代码、未修改正在运行服务的参数。

容器证据：
/workspace/dspark_ctxwin_validation/grouping_sweep.py、
/workspace/dspark_ctxwin_validation/grouping_sweep.json、
/opt/w00958190/log/dspark_grouping_sweep.log。
当前127.0.0.1:8872健康检查200，原后四卡vLLM服务继续运行。
