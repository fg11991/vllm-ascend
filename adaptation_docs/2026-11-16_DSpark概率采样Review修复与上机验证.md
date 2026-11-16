# 2026-11-16 DSpark概率采样Review修复与上机验证

本次核对 `030_naie_dev_greedy_fix` 的 `eb34c66be69ec9ea518b6fd0aac402dbd7914491`。
原始开发基线仍为 `singzhou/030_naie_dev@7f93f07236def39e26b491798c7c461a552c2b23`。

## Review 结论与修复

| 问题 | 核对结果与处理 |
| --- | --- |
| P1 DP dummy 复用真实温度 | 成立。真实调用显式传入 metadata，dummy/profile 的 `None` 不再从 runner 补回。dummy 只走映射后的 argmax，且不导出 q。 |
| P1 同步语法裁剪造成分布偏差 | 成立。保留首个非法候选及其 q；仅在生成 grammar mask 的副本中使用占位，令 p(x)=0 触发正常拒绝与残差恢复。 |
| 动态 K 的 query 长度与缓冲区 | 成立。按当前 K 更新 query 长度、裁剪返回列和动态 confidence，保护 kernel 写入；K=0 返回前清缓存。 |
| 缺失请求或步数导致整批丢 q | 成立，主要影响接受率。保留已有行，缺失行和实际 -1 占位行使用零 q，强制拒绝并从 p 恢复。 |
| 无映射但草稿头比 target 宽 | 成立。额外列在 greedy 和概率采样前均按真实 target 词表裁掉。 |
| mixed batch 的 greedy 行 q | 与上游约定一致，不是当前验证器的分布错误。仅 random 行的 q 表示真实提议分布，greedy 行由独立 mask 验证，不读取该 q。 |
| 逐步 full-vocab 分配与 stack 峰值 | 分配风险成立，尚无此前代码实际 OOM 的证据。草稿在小词表上采样，再把 q 写入最终预分配缓存，减少 full-vocab logits/noise 和 list/stack 重复缓存。 |

语法反例中 `q=(0.4,0.1,0.5)`、约束后的 `p=(0.5,0.5,0)`：旧截断流程输出 `(0.65,0.35,0)`；保留首拒位置后，经 `(p-q)+` 恢复到 p。
修复只对 V1 DSpark、`draft_sample_method=probabilistic`、`rejection_sample_method=standard` 启用，其他模式沿用原行为。

## 关键代码

- `vllm_ascend/spec_decode/llm_base_proposer.py`：metadata 契约、当前 K、缓存清理、小词表采样及最终概率缓存。
- `vllm_ascend/spec_decode/dspark_proposer.py`、`vllm_ascend/spec_decode/utils.py`、`vllm_ascend/ops/triton/spec_decode/utils.py`：dummy、动态 query/confidence 与 kernel 边界。
- `vllm_ascend/worker/model_runner_v1.py::_get_spec_decode_draft_probs`：按请求和步数对齐缓存，缺失与占位行强制拒绝。
- `vllm_ascend/patch/platform/patch_dspark_structured_output.py`：保留首个非法候选，只修改 mask 输入副本，保持 grammar 回滚。
- `tests/ut/spec_decode/test_dspark_probabilistic.py`、`tests/ut/spec_decode/test_dspark_review_regressions.py`、`tests/ut/spec_decode/test_dspark_probability_cache.py`、`tests/ut/patch/platform/test_dspark_grammar_rejection.py`：回归测试。

验证阶段仍需保留完整 target 空间的 q；runner 按请求重排时也会分配对齐后的概率张量。
此次减少草稿采样的冗余分配，不保证任意批量和词表大小下均不会 OOM。

## 上机范围

用户本轮明确授权 `910b3` 的前四卡及停止其占用进程，覆盖此前仅使用后四卡的限制。
源码、配置与测试在 `wyd_va_030rc2_1028` 容器的独立目录内完成；后四卡原有服务继续保留。
配套 vLLM 为 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，使用 V1、eager、同步调度。
该版本在 DP>1 时会禁用按批量调整 K，因此分别验证 DP2×TP2 的静态 K=7 和单 DP×TP4 的动态 K。

## 验证记录

- 本地四个回归文件：68 passed，3 个 NPU 测试因无设备跳过。
- 910b3 同一组测试：**71 passed**。包含真实采样器的概率分布检查、跨步 strided q 写入，以及两种 anchor 模式的 kernel sentinel 边界检查。
- 已暂存的完整代码运行 `bash format.sh ci` 通过。
- 端到端请求、动态 K 的实际步数证据及最终服务健康状态将在完成后补充。
