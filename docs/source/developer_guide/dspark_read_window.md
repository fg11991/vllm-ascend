# DSpark MRv2 context window：先让 draft 只读需要的历史

本文解释第一阶段的“只读窗口”适配。第二阶段的“回收窗口外 KV”见
[KV 保留与分组报告](dspark_kv_retention.md)，设备证据见
[实机验证报告](dspark_npu_validation.md)。

## 这项改动解决什么问题

DSpark 每次并行提出一组候选 token，target 再验证这些候选。
target 仍然需要自己的完整历史；draft 则可能按训练配置，只使用最近一段历史来提出候选。

此前 MRv2 的 draft 输入仍暴露整段历史。
第一阶段做的事，是让 draft attention 看见与配置相符的对齐窗口。
这改变的是 draft 的读取范围；调度器仍然把 draft 当作全量保留的 KV。
因此，“只读约4k”在当时不意味着“只存约4k”。

本文中的“窗口”有两个容易混淆的含义：
读窗口回答“计算需要哪些 token”；保留窗口回答“哪些 KV 块仍不能释放”。
第一阶段只处理前者。

## 窗口如何确定

配置来自 draft checkpoint 的 dflash_config：

- context_window：记作 W，历史窗口长度。
- context_window_align：记作 align，窗口起点向下对齐的粒度。
- B：本次 draft block 长度，本场景等于 num_speculative_tokens。

以 draft 的 seq_len 为准，定义：

```text
anchor = seq_len - B
start  = floor(max(anchor - W, 0) / align) × align
读取范围：[start, seq_len)
```

anchor 是本次 draft block 的起点。窗口包含此前的历史，以及该 block 自身的 B 个位置。
block 内原有双向 attention 语义保持不变。

例如，anchor=10000、W=4096、align=128、B=7：

```text
anchor - W = 5904
向下对齐后 start = 5888

历史：[5888, 10000)，共4112 token
本次draft block：[10000, 10007)，共7 token
最终读取：[5888, 10007)
```

因此不能把它替换成逐 token 移动的4096长度 causal 滑窗。
对齐带来的额外历史是这个定义的一部分，不是多读错误。

## 原来怎么读，现在怎么读

MRv2 用 block table 将逻辑 token 位置映射到 KV 池中的实际块。
它可以类比为页码目录：目录里的数字指向数据，而不是数据本身。

改前，draft 的 metadata 使用完整 block table 和完整长度。
改后，DraftContextWindow 根据每个请求的 start，从完整目录对应的列开始，
收集一份较短的目录副本；长度同时减去 start。
attention 从短目录的第一列开始读取，实际仍指向原来的 KV 数据。

这里没有搬运、压缩或重新计算历史 KV，也没有把 KV 写入改为循环覆盖。
写入继续使用绝对 token 位置和原 slot mapping。
短目录只用于读取，这样读侧优化不会改变 context KV 的写入位置。

align 必须是 kernel block size 的整数倍，否则起点可能落在 kernel page 内，
仅移动 block table 列无法表达这个偏移。本实现启动时检查此约束。
本次 GQA 验证的 kernel block 为128；后续为匹配 GDN 的 scheduler block 为896。
896是调度器分配、保留和回收的单位，一个调度块在worker展开为7个128-token kernel pages。
align=128只需满足读取侧kernel page对齐，不要求它是896的倍数。

## 为什么不只改一个长度

attention 有多条读取长度的路径：设备上的精确 seq_lens、
host 侧的 context 长度上界、MLA/FIA 使用的长度列表，以及批次最大长度。
只改其中一项，会造成“目录很短，但某条路径仍声称有完整历史”的不一致。

适配同时约束这些入口。speculator 在构建 draft metadata 时临时换入窗口目录和长度，
构建结束后恢复原引用；重入保护避免重复减去起点。
非 draft 组的输入保持原样。

窗口目录和工作缓冲区在初始化时一次分配，运行时原位更新。
这样图捕获和回放可以使用稳定的地址，而不是每步替换新 tensor。

## FULL 图捕获还需要单独处理

图捕获没有正常请求的全部长度准备过程。
它曾经用模型最大长度进行 attention tiling，但绑定的却已经是短窗口目录。

举例说，目录只能描述约4k token，却向 kernel 宣称可读取8k甚至更长，
即使普通运行时 metadata 正确，捕获路径仍可能越界。

第二个提交为捕获提供专用的最大长度上限，并同时受实际目录副本容量约束。
短 max_model_len 场景也不能让声明长度超过目录。
目录列数按ceil((W+align+B)/kernel_block_size)+1分配，且不超过原完整表列数；
运行与捕获声明的最大读取长度还受目录实际可表示容量约束。
Qwen GQA 的 FA4 使用设备长度；这个修补尤其涉及 MLA/FIA 的 host 长度路径，
不能因为 GQA 没有报错就认为 MLA 路径也安全。

本次设备请求 FULL 后，target 的 GDN backend 实际使用 FULL_DECODE_ONLY，
draft 捕获 FULL 图。它不代表 target 的全部 prefill 都能运行 FULL 图。

## 配置边界和已知限制

W和align都没有配置时，不启用读窗口；只配置其中一个时明确报错。
DCP/PCP被拒绝，因为当前起点是全局绝对位置，而它们的块表涉及跨卡分片。

接近 max_model_len 时，输入准备会截断 seq_len，由此推导的 anchor 可能偏小。
窗口可能提前开始，读到少量额外的已写历史。
这是第一阶段已有的边界偏差，anchor最多后退B。
第二阶段把这B个位置计入安全余量，见[保留量推导](dspark_kv_retention.md#为什么保留4239而不是直接保留4096)。
回收改动没有重写计算定义。

本次使用 W=4096、align=128 是隔离测试配置。
原 Qwen3.8-27B-DSpark-Base checkpoint 没有这对字段，原文件未被修改。
不能据本实验断言该 checkpoint 的训练窗口一定就是4096/128。

## 代码、提交与证据

| 提交 | 作用 |
| --- | --- |
| 6b90c0a2b | 引入 DraftContextWindow、speculator 接线和图路径输入交换 |
| 4a01608da | 修补捕获长度和短目录容量边界 |
| cc746b7ca | 后续的调度器保留策略；不是本阶段读算法的一部分 |

主要文件位于 vllm_ascend/worker/v2/spec_decode：
dspark/context_window.py、dspark/speculator.py、dflash/aclgraph.py。

原提交时的证明主要来自 CPU：窗口起点黄金样例、目录收集、越界补零、
缓冲复用、host 上界、重入和捕获上限。后续联合回归集合共389项通过，
除上述读窗口检查，还包含回收不变量、分组、FA4/FIA metadata和MLA存储布局；
不能理解为389项全部都是第一阶段专用测试。
GQA 的后续设备测试进一步覆盖 eager、图模式和长请求。
MLA 的布局及 metadata 有 CPU 证据，未用真实 MLA checkpoint 设备验收。

回收实验的基线已经包含这两个读窗口提交。
因此改前改后输出相同，支持的是“加入回收后，保持已有读窗口的计算结果”，
不是“全历史 attention 与窗口 attention 对所有输入都相同”。

## 怎样判断第一阶段是否成功

成功标准是：draft 的每条读取路径使用一致的对齐范围，
原写入位置和 block 内语义保持不变，普通执行与图回放均可运行。
本文设备验收范围是所列GQA模型与请求；MLA路径还需单独设备验证。
它不承诺减少调度器保留的 KV 块，也不能仅凭 npu-smi 判断成功。
真正的旧块回收是下一阶段的工作。
