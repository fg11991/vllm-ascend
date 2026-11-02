# DSpark 窗口外 KV 回收开关

日期：2026-11-02。分支：`030_dspark_ctxwin_tmp`。

## 修改范围

前一提交 `89b0bd3b5` 为开启 APC 删除了 `validate_windowed_retention()` 的两行 prefix-caching 禁令，
同时调整配置 UT 并归档实机证据；没有重写联合前缀命中、窗口读取、KV 回收或尾块保护策略。

本次新增 `speculative_config.dspark_drop_outside_window_kv`，默认 `true`，严格接受 JSON 布尔值。
配置注册在现有 Ascend speculative-config 补丁内，保持 upstream 类的身份，重建独立和嵌套的 Pydantic schema，
不需要修改另一份 vLLM checkout，也不写入权重配置文件。

`false` 时不转换 draft KV spec；窗口初始化允许原 FullAttentionSpec 分配，并继续使用同一套窗口 metadata。
`true` 时保持已有回收和分组丢失保护。窗口数学、原始块表、target 缓存策略及其他启动限制均保持。
配置值加入已有启动日志，方便检查实际选择。完整使用方式见 [回收说明](dspark_kv_retention.md#独立控制窗口外-kv-回收)。

## 验证

在 `910b3` 的 `wyd_va_030rc2_1028` 容器创建独立源码副本：
`/workspace/dspark_ctxwin_validation/kv_drop_toggle_20261102_713omnje/source`。
配套 vLLM 为 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，工作区无改动。
测试进程显式从这个副本导入 Ascend；未修改已有服务源码或安装环境。

以下五个文件合计 **154 passed, 14 warnings in 8.11s**：

- `tests/ut/worker/v2/test_dspark_context_window_retention.py`
- `tests/ut/worker/v2/test_dspark_context_window.py`
- `tests/ut/worker/v2/test_dspark_speculator_context_window.py`
- `tests/ut/patch/platform/test_patch_speculative_config_dspark.py`
- `tests/ut/patch/platform/test_patch_speculative_config_dflash.py`

覆盖默认值、布尔值拒绝、CLI JSON schema、嵌套配置校验、序列化/复制/pickle，
以及 APC 开关与回收开关组合、关闭回收后的原 spec 保留、共享 full-attention 组的窗口读取。
实际回归用例核对 anchor=996 时仍从 640 读到 1000，读长度 360，退出作用域后原始块表及长度不变。
这些是 CPU 逻辑 UT，在真实容器依赖环境中运行，不是 NPU 推理验收。

另以实际 Qwen3.8-27B target、窗口增训 draft 路径初始化完整 ModelConfig/SpeculativeConfig，
未 mock post-init；`true`、`false` 都成功且窗口仍为 2048/128。此检查只读取模型配置，不加载权重执行推理。
全仓 `bash format.sh ci` 的 18 个 manual hooks 通过。
仅为配置跨进程序列化回归测试增加 pickle 导入白名单；测试只反序列化自己构造的数据，生产代码没有新增 pickle 操作。

日志、源码 SHA256 和检查汇总位于 [检查记录](dspark_kv_drop_20261102_evidence/checks.json)，
包括 [UT 输出](dspark_kv_drop_20261102_evidence/ut.txt)、
[实际模型配置检查](dspark_kv_drop_20261102_evidence/config_smoke.txt) 和
[全仓格式检查](dspark_kv_drop_20261102_evidence/format_ci.txt)。

## 验证边界

2026-10-31 的 14 请求、59136 缓存 token 命中仅属于回收开启的历史配置。
本次未重跑完整 NPU 推理矩阵，不能据此宣称回收关闭时的 APC 命中率、图运行、输出等价或性能已实机验证。
现有前台留驻服务未重启、停止或切换。全仓 UT、其他窗口、抢占/驱逐压力、MLA、PD、DCP/PCP 未新增验收。
