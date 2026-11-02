# DSpark 固定 KV 分组的回收隔离诊断证据

结论、精度限制与未覆盖场景见 [诊断报告](../2026-11-02_dspark_retention_isolation.md)。
本目录归档 2026-11-02 已完成的 20 次请求；本次整理没有重跑 NPU 推理或检查当前服务。
20 次请求只使用 **1 条独立长 prompt**，不是 20 道精度题。

## 文件

- `final_summary.json`：全部 20 次请求的计数、分阶段汇总、同缓存成对输出比较及首次回收位置。
- `runtime_manifest.json`：模型、源码指纹、依赖版本、固定配置和收尾模式。
- `raw_evidence.tar.gz`：原字节复制的 134 文件证据包；不含权重、venv 或编译缓存。
- `bundle_manifest.json`：归档 SHA256、大小、来源和内部索引。
- `integration_checks.json`：本次离线复核及格式检查记录，不与实机推理结果混算。
- `secret_scan_review.json`、`secret_scan.log`：解压内容扫描及两个输入 token 哈希误报的复算记录，未修改扫描规则。
- `format_ci.log`：本次全仓 `bash format.sh ci` 的最终通过输出。

归档含 20 条请求的 payload、输出 token IDs/文本、metrics、逐步接受数、回收/保留事件，
以及固定输入、模型配置、启动和客户端脚本、服务日志、健康快照与 NPU 状态。
`evidence/previous_service_exit.log` 是本轮连接前上轮服务的退出日志，不属于本轮推理失败。
归档内“没有 commit/push”指实机实验阶段；本次提交仅整理记录。

## 离线复核

先按 `bundle_manifest.json` 核对归档 SHA256，解压到新目录，进入
`retention_isolation_20261102_round02` 后执行：

```bash
shasum -a 256 -c SHA256SUMS
python3 summarize_final.py
```

第二条命令只重算已有 JSON，不连接 NPU 服务；会覆盖解压副本内的 `final_summary.json`。
本次整理已校验 133 项文件哈希，并确认重算摘要与归档完全相同。
已核对 20 个 payload 的 temperature=0、top_p=1、seed=42、max_tokens=512、输入长度 14512，
以及关键源码哈希与当前 tmp 分支一致。

实机入口 `serve.sh`、`run_isolation.py`、`run_cross.py`、`run_reverse.py` 含原实验路径和服务地址。
如需复跑，应另建隔离目录并核对服务归属、卡占用和输出路径；不要原样覆盖旧结果。
这些实验脚本不是新增的生产功能、自动化 ST 或通用基准工具。

## 解释边界

两种模式始终保持生产配置 `dspark_drop_outside_window_kv=true`、APC=true、固定 9 个 KV group。
实验钩子在正常窗口块退休与跳过退休之间切换；跳过时同时保留块表和引用计数。
因此并未进一步独立分离块表置空与物理引用释放。

全部热请求汇总下降 1.2781 个百分点，但部分输出轨迹/长度不同；
两对完整 418-token 输出一致的同缓存对照，方向相反。
反向重复及相同模式自重复仍出现差异，不能把汇总接受率差值当作已定位的因果效应。
相同缓存块 ID 不替代设备 KV 内容逐元素验证，greedy 也未保证逐 token 确定性。

未验证用户私有 checkpoint/数据集、任务答案正确率、多并发、高缓存占用下驱逐/新槽复用、
长时间稳定性、PD、DCP/PCP 或完整 draft logits。CPU 审计可能改变时序，记录不作为 TPOT 评测。
服务状态和原有 NPU Alarm 均为当时快照，不是本次提交时的健康认证。
