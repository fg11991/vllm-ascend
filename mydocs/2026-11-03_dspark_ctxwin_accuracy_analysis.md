# 2026-11-03 DSpark context window 与 KV 回收的精度保持分析

## 结论

**当前业务样本未显示明确、系统性的答案精度下降；这一现象与代码设计一致：draft 负责提案，target 负责验证，窗口外 KV 回收不应改变有效 attention 读取。**
工程上可将本轮结果记录为“未观察到明确精度退化”，不能将输出一致率直接当正确率，也不能扩大为任意模型、并发和缓存状态下严格无损的证明。

本次只复核已有 JSON、历史实机证据及源码，没有重新运行 NPU 测试。源码核对基于 `030_dspark_ctxwin_tmp` 的 `26dd36565`。

## 本轮实验与结果

数据来自 `fg11991/dspark_repository` 提交 `334a50e961401d8b801357110aa1c621d288036d`：

- [final_comparison.json](https://github.com/fg11991/dspark_repository/blob/334a50e961401d8b801357110aa1c621d288036d/qwen38-27b/comparison/final_comparison.json)。
- [tool_comparison.json](https://github.com/fg11991/dspark_repository/blob/334a50e961401d8b801357110aa1c621d288036d/qwen38-27b/comparison/tool_comparison.json)。
- 比较实现：[agent_benchmark.py](https://github.com/fg11991/dspark_repository/blob/334a50e961401d8b801357110aa1c621d288036d/Magistral_Eagle3_Project/benchmark/agent_benchmark.py) 的 `classify()` / `compare()`。

报告按 `nonctx_*.jsonl` 与 `ctx_*.jsonl` 配对。每组 50 道最终回答、50 道工具调用，共 100 对、200 次请求。
逐项核对两侧 `req_js` 完全相同；模型请求名 `qwen3.8-27b`，并发 1，temperature=0、top_p=1、seed=42、max_tokens=1024。
它们保存了请求与输出，但没有完整服务 manifest，不能单凭文件名确认实际 draft checkpoint、APC、drop 和 KV 分组的唯一变量；也不能用请求模型名确认权重身份。

| 类型 | 输出一致 | 输出不同，待审阅 | 无法判定 | 总数 |
| --- | ---: | ---: | ---: | ---: |
| 最终回答 | 36（72%） | 11 | 3 | 50 |
| 工具调用 | 42（84%） | 8 | 0 | 50 |

### 最终回答：多数差异不涉及核心结论

11 条差异中，9 条主要为措辞、排版或后续建议变化，人工复核未见核心结论明显冲突。例如：

- `2c396e53`：两侧均查询到 95 台虚拟机，CPU、内存、磁盘范围一致，只是段落和措辞变化。
- `807acf47`：两侧均为 3 条可用北向连接，IP 与端口一致，描述顺序变化。
- `d99a44a4`：告警说明仅有标题末尾冒号差异。
- 其余主要为“未查到”的后续建议变化；这不证明新增建议已经通过业务验证。

另有两条行动决策分歧，应保留为复核项，不能算作已确认的同义改写：

| req_id 中的会话前缀 | non_ctx | ctx | 解读 |
| --- | --- | --- | --- |
| `2d78bb61` | 继续调用工具查询 CFG 表索引 | 返回表信息并结束 | 可能提前结束，未给出索引详情；把统计列名解释成实际字段也有风险，工具结果自身存在表头歧义 |
| `562ec1e7` | 汇总已有的 96 个网卡端口查询结果 | 再次发出与历史相同的查询 | 已观察到行动变化；本轮没有最终回答，不能由单步结果判断后续是否恢复或形成循环 |

3 条无法判定的会话前缀为 `343e19fc`、`ab7bd6f0`、`de16199d`：两侧均在 1024 token 处 `finish_reason=length`，没有请求错误。它们属于输出预算不足，不能算成 ctx 单侧失败。
36 条一致中还有 2 条是两侧均继续调用工具，因此“finalanswer 文件里的 complete”只表示这次响应完整，不等于整个 Agent 任务完成。

### 工具调用：8 条差异都是参数键名变化

8 条全部调用相同的 `Skill` / `cn-data-query`，查询文本相同，唯一差异是 `args.query` 与 `args.question`：6 条由 query 变为 question，2 条反向变化。
仅作为诊断，将两个键视为别名后，50 对结构化输出全部一致；没有发现工具选错或查询文本改变。

但报告中的 Skill schema 只规定 args 是可带任意字段的对象，并未保证执行器支持两种键。
如果实际 Skill 同时接受这两个键，这 8 条不构成业务差异；如果只接受其中一个，就有参数丢失风险。
本轮没有执行生成的工具调用，不能将上述诊断统计改写成“工具业务成功率 100%”。

### 本轮如何理解“精度 OK”

这里指：核心答案和工具意图总体保持，没有看到大面积内容错误或明确的系统性退化。
不把 72%/84% 的一致率解读为准确率，更不能据此说精度下降 28%/16%；同样不把两个行动分歧抹掉。
两组均为 `self_repeat_checked=false`，只有一次运行，尚不能将差异分离为配置影响或同配置运行波动。

## 从代码解释为什么可以保持精度

### 1. draft 改变提案效率，最终 token 仍经过 target 验证

MRV2 的接入见 [patch_triton.py](../vllm_ascend/patch/worker/patch_v2/patch_triton.py)，它将上游 rejection sampler 接到 Ascend 实现。
[rejection_sampler_utils.py](../vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py) 的正常 greedy 分支执行：

```python
accepted &= target_argmax == draft_sampled
tl.store(sampled_ptr + req_idx * sampled_stride + i, target_argmax)
```

draft 提案与 target 选择一致才接受；首次不一致时保留 target 选择并终止该段继续接受。全接受后的 bonus token 也按 target 分数选择，见 [resample.py](../vllm_ascend/ops/triton/v2/spec_decode/resample.py)。
这里讨论正常验证路径，不包括合成接受率模式 `SYNTHETIC_MODE`。

因此，训练或读窗口改变 draft 提案，首先可能改变接受率和执行步数；**接受率下降不等于最终答案正确率下降**。
在相同历史下 target 分数及验证状态正确且一致，是 greedy 输出保持一致的前提。实现存在错误或 target 数值发生变化时，不能用这一算法原理替代实机校验。

### 2. 少读上下文与释放已不再读取的 KV 是两件事

[context_window.py](../vllm_ascend/worker/v2/spec_decode/dspark/context_window.py) 的 `window_start()` / `DraftContextWindow` 定义：

```text
start = floor(max(anchor - W, 0) / align) × align
draft 读取 [start, seq_len)，其中包含本轮 draft block。
```

仅对 draft group 构造窗口 block table；KV 写入继续使用原始绝对位置 slot mapping。
[cache_retention.py](../vllm_ascend/worker/v2/spec_decode/dspark/cache_retention.py) 的 `configure_draft_kv_retention()` 只转换模型声明的 draft paged attention 层的 KV spec，交给已有滑窗 manager 回收；不改 target 的保留策略，不改 FA4/FIA 的 attention mask 属性。

相同读窗口下，回收的是以后不应再访问的历史块，设计目标是不改变有效计算输入。
“全上下文 draft 改为窗口 draft”则确实改变提案模型的输入，不能声称两者 draft logits 相同；最终答案保持依赖 target 验证。

### 3. 安全余量覆盖对齐、投机回退与异步在途步骤

`retention_window()` 设置 `R = W + align + B + S + 1`。W=2048、align=128、B=S=7 时 R=2191。
历史配套 vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607` 的 `KVCacheManager.allocate_slots()` 按扣除 in-flight 的已处理长度 C 回收。
在已核对的 `C - anchor <= B + S` 条件下：

```text
C - R + 1 <= anchor - W - align < start
```

`SlidingWindowManager.remove_skipped_blocks()` 仅释放该边界之前的完整调度块；覆盖有效读起点的块仍驻留。
对应源码为配套 vLLM 的 `vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/single_type_kv_cache_manager.py`；此处依据历史源码审计，未重新读取远端工作区。
边界回归见 [test_dspark_context_window_retention.py](../tests/ut/worker/v2/test_dspark_context_window_retention.py)，实机证据见下文。

### 4. 分组独立不是 APC、多并发正确性的全部理由

target 与 draft 的保留策略可独立，但它们仍共享物理块池，hybrid coordinator 还会协调共同前缀命中长度。
APC 必须同时满足所需 draft 尾块驻留、各组共同命中合法；多请求共享块时，每次回收只释放本请求引用，仍有引用的块不能作为空闲新槽覆盖。
关闭 `dspark_drop_outside_window_kv` 会保留原 KV spec，本模型可能由 9 组退回 4 组；它不是天然的“固定布局只切释放”实验。

解除 APC 启动限制只删掉配置拒绝，不重写联合命中或 draft 缓存恢复策略。源码设计提供解释，联合命中与引用计数是否实际正确仍需设备证据。

## 与历史实机证据的关系

| 证据 | 支持什么 | 不能扩大为什么 |
| --- | --- | --- |
| [APC/drop 四组验证](2026-11-02_dspark_apc_drop_validation.md)：84/84 完成，32/32 取值回答正确；其中只有 4 种不同题目 | 窗口外关键事实仍可由 target 使用；APC+drop 可共存 | 不是 32 道独立精度题，也不是通用正确率 |
| 同轮 APC+drop 命中 145152 token，55 次块引用释放，错峰并发 4 覆盖共享引用递减 | 真实命中、尾块驻留及共享引用管理的有限实机证据 | 不代表 APC 驱逐与物理块复用压力已覆盖 |
| [同分组隔离](2026-11-02_dspark_retention_isolation.md)：20/20 完成，16 次热请求各命中 14336 token | 没有观察到接受率差异稳定跟随实际回收；首次交叉两对输出均为相同 418 token | 只有一个 prompt，不能否定私有数据集约 5% 的接受率下降 |
| [窗口增训补充实验](2026-10-31_dspark_trained_window_followup.md)：同组不回收也出现 target 候选排序变化和输出分叉 | greedy 不保证当前执行路径逐 token 确定；回收不是分叉的必要条件 | 未定位全部算子原因，不能将新报告所有差异都归为浮点噪声 |

本轮报告与这些历史实验不是同一次运行，也没有完整运行身份可证明它们使用相同 checkpoint；因此只作交叉解释，不合并成一个正确率。

## 保留边界与复核入口

- 本轮无标准答案、target-only 基准或同配置自重复；不能给出数值化正确率、统计显著性或严格等价证明。
- 两条行动分歧、Skill 参数键名兼容性、三条截断题仍可复核；未验证真实工具执行后的多轮任务成功率。
- 本轮并发为 1，不能代替新模型的多并发验收；历史并发 4 只覆盖指定模型和取值/共享缓存场景。
- 未验证 APC 高压力驱逐与回收块再分配组合、长期稳定性、真实 128k 输入、MLA 实机、PD、DCP/PCP。
- 接近 max_model_len 时，读窗口 anchor 的截断会使窗口起点略提前；实现已留驻安全余量，但不能称为所有位置与训练定义严格一致。

原始两组 JSONL 位于报告记录的 `/dpc/hot/w00958190/101_workspace/dspark/test_output/accuracy/ctx_compare_20261102_120449/`。
上述两个 GitHub JSON 的 SHA256 分别为：

```text
final_comparison.json 341d23b7a5afb63233c745a3b81d0193e48190a4c15cb72fa0c7aa2df8479a2d
tool_comparison.json  f79da05ece7b42cba36221b82a0841ee1bc76283ad4195069d065b0fa6e0f6cb
```

全阶段结论见 [适配验证附录](../adaptation_docs/2026-11-03_DSpark窗口KV适配验证附录.md)。
