# DSpark APC 实机证据索引

结论与限制见 [实机报告](../2026-10-31_dspark_apc_validation.md)。
本目录整理 2026-10-31 已完成的实验，不将记录整理当作重新进行 NPU 测试。

- `validation_results.json`：原始中文汇总快照；其“本轮未修改 tmp 工作区”和留驻服务状态均指实验当时。
- `results_summary.json`：逐请求结果、接受长度、cache lookup、回收计数与输出 token 比较。
- `source_audit_final.json`：原目录、隔离副本以及 vLLM 的最终源码核对。
- `retained_service_health.json`：实验结束时的健康检查及留驻进程，不能代表当前状态。
- `integration_checks.json`：2026-11-02 整理提交时的检查、环境修复及剩余测试边界。
- `format_ci.log`：补齐工具依赖后的全仓 `bash format.sh ci` 通过记录。
- `raw_evidence.tar.gz`：78 个原始文件，按原字节保存。
- `raw_evidence_sha256.json`：上述 78 个文件在归档内的路径及 SHA256。
- `bundle_manifest.json`：归档 SHA256、来源目录、实验与整理日期。

归档中的 `evidence/` 包含全部 14 条请求的 payload、输出 token IDs、metrics 快照、
`off/on.scheduler.jsonl`、`off/on.startup_audit.json`、模型 manifest、固定输入、
`unit_test.log`、设备健康记录与 `logs/` 下的完整服务及客户端日志。
`run_requests.py`、`apc_audit.py`、`startup_audit.py`、`summarize.py` 和 `serve.sh` 保留当时验证入口。
脚本带有原实验绝对路径；它们用于复核，复跑前需创建新的隔离目录并调整路径、输出标签。

解压到新目录后，可按 `raw_evidence_sha256.json` 对每个文件重新计算 SHA256。
汇总可用归档内的 `evidence/summarize.py` 根据同目录请求结果、metrics 和 scheduler 记录重新生成。
归档包含初次客户端输入序列化失败日志；它不是服务端 APC 失败。
原始汇总对尾块保护回退的说明不是运行时分支观察，具体限制以实机报告为准。
