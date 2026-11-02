# DSpark APC 与回收开关四组实机证据

结论、精度限制和未覆盖场景见 [验证报告](../2026-11-02_dspark_apc_drop_validation.md)。
本目录整理2026-11-02已完成的84个请求，不表示重新运行NPU推理，也不重新声明服务当前健康。

- `final_analysis.json`：四组成功数、实际命中、回收/共享引用、接受长度和首个输出分歧摘要。
- `runtime_manifest.json`：实验提交、依赖版本、关键源码SHA256与采样设置。
- `raw_evidence.tar.gz`：完整验证包，按实验结束后的原字节复制，共251个文件。
- `bundle_manifest.json`：归档SHA256、大小、来源和归档内文件索引位置。
- `integration_checks.json`：本次文档整理的复核结果；不与实机结果混算。
- `format_ci.log`：本次全仓格式检查的最终通过记录。

## 归档内容

`SHA256SUMS`列出其余250个文件的哈希。原始报告和脚本保持当时内容，
其中“没有新增commit/push”和留驻服务状态均指实验阶段，本次提交只增加报告及证据。

归档包含全部84条请求的payload、输出token IDs、文本、output logprobs和完成原因，
各阶段metrics、A/B/C/D的scheduler事件、固定`fixtures.json`、模型配置副本、
源码/依赖manifest、二进制链接清单、服务快照、NPU状态及完整客户端/服务日志。
首次启动缺少自定义算子二进制的完整异常、B组主动关停后的异常也一并保留。

`serve.sh`、`run_matrix.py`、`apc_audit.py`、`startup_audit.py`保存实验入口与CPU观察逻辑。
这些脚本带原实验绝对路径；复跑须新建隔离目录、修改输出标签，并核对服务归属及空闲状态。
不能原样运行服务切换脚本覆盖旧结果或终止其他任务。此次整理不运行这些实机脚本。

## 离线复核

先按`bundle_manifest.json`核对归档SHA256，再解压到新目录。
在解压目录内执行以下命令只重新计算已有数据，不连接NPU服务：

```bash
shasum -a 256 -c SHA256SUMS
python3 summarize_round.py
python3 finalize_analysis.py
```

汇总脚本会重写该解压目录内的派生JSON，因此应先保留解压出的原始摘要用于比较。
本次整理已逐文件校验250项SHA256，并离线重算汇总，核对84条请求和采样参数。
原始归档字节不随复核而修改。

D组22次lookup对应21条请求。同一请求重复查询一次，原始事件命中求和为155008；
按request_id去重后为145152，与metrics实际命中增量一致。
`results_summary.json`的`lookup_event_hit_tokens`不能当作实际复用量；
应使用`request_hit_tokens`并与metrics核对。分支外露的`final_analysis.json`中`hits`为实际命中量。

本轮4种不同取值题重复得到32/32正确，不是通用精度基准。
C/D仅15/21条输出逐token一致；近似并列候选是解释性证据，不是分歧根因已完全定位。
D组覆盖共享引用下回收，未覆盖APC下已回收块的驱逐、哈希失效和物理再分配组合。
归档内脚本不是新增的自动化ST，格式检查也不替代功能测试。
