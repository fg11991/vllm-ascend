2026-10-31

# 补充实验的便携证据

- 两张CSV分别保留500次业务请求与30个正式性能条件的原始计数、实际配置和日志路径。
- `formal_per_request_results.jsonl.gz`含620条逐请求结果、输出token IDs与输入token散列；为避免重复存储，完整请求模板与输入IDs仍在远端原始结果。
- JSON摘要提供分叉、复用、分组和健康检查证据；不存在的待完成项目不会凭空填入。
- `validation_harness.tar.gz`保留本轮验证脚本、隔离模型配置与首次设备异常日志，不含模型权重、编译缓存或生产环境。
- `remote_results_sha256.json`列出容器原始结果的完整路径、大小和SHA256。完整设备轨迹位于`/opt/w00958190/log/dspark_20261031_diag_*.jsonl`，不计入正式速度。

所有正式结果根目录为`/workspace/dspark_ctxwin_validation/round_20261031_isolated`。先阅读正文与技术附件的范围和限制；探针检查通过不等于输出严格等价已被证明。

仓库副本仅将两张CSV换行为LF，数据逐字段一致。`remote_bundle_manifest.json`保留下载原包散列，`bundle_manifest.json`记录最终交付文件散列；交付时新增读者检查与格式检查记录。
