# Qwen3.8-27B DSpark 上机验证：技术附录

实验日期：2026-10-30。本文为 [实验结果说明](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证结果.md) 的复现材料。配置均使用完整名称；原始文件名作为证据标识保留。

## 配置、代码与模型目录

本次目标模型固定为 `/opt/models/Qwen3.8-27B`。原代码为 `fbc3e6b8ab5d516ae8094b4ff3bf2929987aaf16`；窗口分支为 `4a01608da7be1950eebc9902e3692d257f304759`，包含功能提交 `6b90c0a2b`。推理源码、原脚本和原始导出没有因验证新增修改；没有新提交、push 或 PR。

| 草稿模型／训练方式 | MRV版本 | 执行模式 | 推理窗口 | 推理代码 | 正式请求数 |
| --- | --- | --- | --- | --- | --- |
| 不开窗增训草稿 | MRV2 | 图模式 | 不开窗 | 原代码 fbc3e6b8a | 100 |
| 不开窗增训草稿 | MRV2 | 图模式 | 不开窗 | 窗口分支 4a01608da | 100 |
| 窗口增训草稿 | MRV2 | 图模式 | 2048／128对齐 | 窗口分支 4a01608da | 100 |
| 窗口增训草稿 | MRV2 | 图模式 | 不开窗 | 窗口分支 4a01608da | 100 |
| 不开窗增训草稿 | MRV2 | 图模式 | 2048／128对齐 | 窗口分支 4a01608da | 100 |
| 窗口增训草稿 | MRV2 | 图模式 | 128／128对齐 | 窗口分支 4a01608da | 100 |
| 窗口增训草稿 | MRV2 | eager模式 | 2048／128对齐 | 窗口分支 4a01608da | 100 |

| 用途／模型配置 | 服务器目录 |
| --- | --- |
| 窗口增训草稿，原始2048／128配置 | /opt/w00958190/specforge_tmp/export_A_agent_ctxwin2048_step50 |
| 不开窗增训草稿，原始完整上下文配置 | /opt/w00958190/specforge_tmp/export_B_agent_nowindow_step50 |
| 窗口增训草稿，移除窗口字段的副本 | /opt/w00958190/specforge_tmp/ctxwin_variants/A_full |
| 不开窗增训草稿，补入2048／128窗口字段的副本 | /opt/w00958190/specforge_tmp/ctxwin_variants/B_w2048 |
| 窗口增训草稿，改为128／128的副本 | /opt/w00958190/specforge_tmp/ctxwin_variants/A_w128 |

变体只复制 config.json 并修改窗口字段，权重用软链；原件哈希及软链目标在原件校验记录中。所有推理服务都使用 TP4、物理卡4–7、推测步数7、max_num_seqs=8、max_model_len=32768、gpu_memory_utilization=0.9、enable_thinking=false、异步调度及前缀缓存。除比较旧、新代码的基线组外均运行窗口分支。

软件包版本：

```json
{
  "vllm": "0.30.0+empty",
  "vllm-ascend": "0.1.dev6051+gfbc3e6b8a",
  "torch": "2.10.0+cpu",
  "torch-npu": "2.10.0.post7",
  "transformers": "5.14.1"
}
```

editable 导入路径为 `/opt/w00958190/vllm-ascend/vllm_ascend/__init__.py`。发行包元数据仍带安装时的旧提交号；实际源码提交由 git HEAD、导入路径及窗口启用日志共同核对。本次没有安装或升级依赖。

## 计数和计时口径

- 草稿接受率 = accepted_draft_tokens / draft_tokens × 100%。它衡量草稿 token 的接受情况，不是题目正确率。
- 平均接受长度（SpecLen）= 1 + accepted_draft_tokens / num_spec_steps。
- 原始计数取每次评测前后独占服务的累计计数差；预热请求在基线采集之前完成。
- 首输出等待时间（TTFT）是每题首个可见内容包延迟的平均值。
- 七套正式 agent 配置的 TPOT 保持历史口径：首个内容包之后的可见流包间隔总和 /（输出 token 总数−请求数）。首包可能含多个 token，因此这是客户端近似口径。
- 长上下文主 TPOT = 首个内容包至完整流结束的时长总和 /（输出 token 总数−请求数），覆盖末尾不可见特殊 token；同时保留旧可见包口径供复核。

长上下文客户端使用原客户端副本，只新增 stream_elapsed、decode_end_to_end_latency 两个字段，没有改变请求体。预备阶段有8个旧口径请求，独立保存在原始证据目录中，不纳入32个正式长上下文请求。

当前环境的缓存重置接口返回404，故取消额外重置；不开窗长组此前跑过预备请求，两种配置的初始前缀缓存状态未完全控制。长组TTFT仅记录、不作直接优劣比较；主要比较相同输入与输出长度下的解码TPOT。

接受率比较基于各配置自然生成的输出，未控制固定目标token轨迹；同一权重的开关窗口对照仍包含生成轨迹变化的影响。

所有阶段之间已核对请求内容哈希一致；长组相同长度下也核对了请求体一致。窗口按训练公式 `start=floor(max(anchor−W,0)/128)×128` 计算，草稿仍按绝对位置写KV。当前块和对齐余量在名义窗口之外；目标模型保留完整上下文。

## 七套配置的完整指标

### 工具调用：接受率和时延

| 草稿训练方式 | MRV版本 | 推理窗口 | 执行模式 | 推理代码 | 有效题数 | 接受率 | SpecLen | TTFT ms | TPOT ms | 平均输出token |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 不开窗增训草稿 | MRV2 | 不开窗 | 图模式 | 原代码 fbc3e6b8a | 50 | 80.4152% | 6.6291 | 2245.138 | 14.438 | 68.34 |
| 不开窗增训草稿 | MRV2 | 不开窗 | 图模式 | 窗口分支 4a01608da | 50 | 80.7787% | 6.6545 | 1292.361 | 14.557 | 68.34 |
| 窗口增训草稿 | MRV2 | 2048／128对齐 | 图模式 | 窗口分支 4a01608da | 50 | 80.4152% | 6.6291 | 1232.865 | 11.738 | 68.34 |
| 窗口增训草稿 | MRV2 | 不开窗 | 图模式 | 窗口分支 4a01608da | 50 | 81.1451% | 6.6802 | 1181.058 | 12.454 | 68.34 |
| 不开窗增训草稿 | MRV2 | 2048／128对齐 | 图模式 | 窗口分支 4a01608da | 50 | 74.0005% | 6.1800 | 1198.602 | 12.646 | 68.34 |
| 窗口增训草稿 | MRV2 | 128／128对齐 | 图模式 | 窗口分支 4a01608da | 50 | 74.1582% | 6.1911 | 1107.627 | 14.319 | 68.34 |
| 窗口增训草稿 | MRV2 | 2048／128对齐 | eager模式 | 窗口分支 4a01608da | 50 | 80.0000% | 6.6000 | 1508.683 | 32.799 | 68.96 |

### 工具调用：原始计数

| 草稿训练方式 | MRV版本／执行模式 | 推理窗口 | 推理代码 | num_spec_steps | draft_tokens | accepted_draft_tokens |
| --- | --- | --- | --- | --- | --- | --- |
| 不开窗增训草稿 | MRV2／图模式 | 不开窗 | 原代码 fbc3e6b8a | 523 | 3661 | 2944 |
| 不开窗增训草稿 | MRV2／图模式 | 不开窗 | 窗口分支 4a01608da | 521 | 3647 | 2946 |
| 窗口增训草稿 | MRV2／图模式 | 2048／128对齐 | 窗口分支 4a01608da | 523 | 3661 | 2944 |
| 窗口增训草稿 | MRV2／图模式 | 不开窗 | 窗口分支 4a01608da | 519 | 3633 | 2948 |
| 不开窗增训草稿 | MRV2／图模式 | 2048／128对齐 | 窗口分支 4a01608da | 561 | 3927 | 2906 |
| 窗口增训草稿 | MRV2／图模式 | 128／128对齐 | 窗口分支 4a01608da | 560 | 3920 | 2907 |
| 窗口增训草稿 | MRV2／eager模式 | 2048／128对齐 | 窗口分支 4a01608da | 530 | 3710 | 2968 |

### 最终回答：接受率和时延

| 草稿训练方式 | MRV版本 | 推理窗口 | 执行模式 | 推理代码 | 有效题数 | 接受率 | SpecLen | TTFT ms | TPOT ms | 平均输出token |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 不开窗增训草稿 | MRV2 | 不开窗 | 图模式 | 原代码 fbc3e6b8a | 50 | 64.7032% | 5.5292 | 1414.762 | 17.761 | 137.40 |
| 不开窗增训草稿 | MRV2 | 不开窗 | 图模式 | 窗口分支 4a01608da | 50 | 65.4064% | 5.5784 | 1445.709 | 16.970 | 140.06 |
| 窗口增训草稿 | MRV2 | 2048／128对齐 | 图模式 | 窗口分支 4a01608da | 50 | 66.6554% | 5.6659 | 1440.241 | 16.813 | 142.34 |
| 窗口增训草稿 | MRV2 | 不开窗 | 图模式 | 窗口分支 4a01608da | 50 | 67.6080% | 5.7326 | 1541.986 | 18.781 | 137.32 |
| 不开窗增训草稿 | MRV2 | 2048／128对齐 | 图模式 | 窗口分支 4a01608da | 50 | 61.3826% | 5.2968 | 1362.893 | 18.037 | 137.34 |
| 窗口增训草稿 | MRV2 | 128／128对齐 | 图模式 | 窗口分支 4a01608da | 50 | 45.6730% | 4.1971 | 1287.885 | 20.883 | 138.52 |
| 窗口增训草稿 | MRV2 | 2048／128对齐 | eager模式 | 窗口分支 4a01608da | 50 | 67.7457% | 5.7422 | 1540.691 | 46.398 | 139.24 |

### 最终回答：原始计数

| 草稿训练方式 | MRV版本／执行模式 | 推理窗口 | 推理代码 | num_spec_steps | draft_tokens | accepted_draft_tokens |
| --- | --- | --- | --- | --- | --- | --- |
| 不开窗增训草稿 | MRV2／图模式 | 不开窗 | 原代码 fbc3e6b8a | 1249 | 8743 | 5657 |
| 不开窗增训草稿 | MRV2／图模式 | 不开窗 | 窗口分支 4a01608da | 1262 | 8834 | 5778 |
| 窗口增训草稿 | MRV2／图模式 | 2048／128对齐 | 窗口分支 4a01608da | 1263 | 8841 | 5893 |
| 窗口增训草稿 | MRV2／图模式 | 不开窗 | 窗口分支 4a01608da | 1204 | 8428 | 5698 |
| 不开窗增训草稿 | MRV2／图模式 | 2048／128对齐 | 窗口分支 4a01608da | 1304 | 9128 | 5603 |
| 窗口增训草稿 | MRV2／图模式 | 128／128对齐 | 窗口分支 4a01608da | 1659 | 11613 | 5304 |
| 窗口增训草稿 | MRV2／eager模式 | 2048／128对齐 | 窗口分支 4a01608da | 1218 | 8526 | 5776 |

## 输出一致性：逐题记录与辅助分析

完整输出由 prefill_token + decode_raw_texts 拼接，包含工具函数名和参数，忽略随机调用ID。对应题目的请求体逐一核对相同。所有不同文本全文保留在原始汇总的 output_comparisons[].different 中。

| 任务 | 比较配置一 | 比较配置二 | 完全相同题数 | 一致率 |
| --- | --- | --- | --- | --- |
| 工具调用 | 不开窗增训草稿，MRV2，图模式，不开窗，原代码 fbc3e6b8a | 不开窗增训草稿，MRV2，图模式，不开窗，窗口分支 4a01608da | 48/50 | 96.00% |
| 最终回答 | 不开窗增训草稿，MRV2，图模式，不开窗，原代码 fbc3e6b8a | 不开窗增训草稿，MRV2，图模式，不开窗，窗口分支 4a01608da | 33/50 | 66.00% |
| 工具调用 | 窗口增训草稿，MRV2，图模式，2048／128对齐，窗口分支 4a01608da | 窗口增训草稿，MRV2，图模式，不开窗，窗口分支 4a01608da | 45/50 | 90.00% |
| 最终回答 | 窗口增训草稿，MRV2，图模式，2048／128对齐，窗口分支 4a01608da | 窗口增训草稿，MRV2，图模式，不开窗，窗口分支 4a01608da | 37/50 | 74.00% |
| 工具调用 | 窗口增训草稿，MRV2，图模式，2048／128对齐，窗口分支 4a01608da | 不开窗增训草稿，MRV2，图模式，2048／128对齐，窗口分支 4a01608da | 47/50 | 94.00% |
| 最终回答 | 窗口增训草稿，MRV2，图模式，2048／128对齐，窗口分支 4a01608da | 不开窗增训草稿，MRV2，图模式，2048／128对齐，窗口分支 4a01608da | 36/50 | 72.00% |
| 工具调用 | 窗口增训草稿，MRV2，图模式，2048／128对齐，窗口分支 4a01608da | 窗口增训草稿，MRV2，图模式，128／128对齐，窗口分支 4a01608da | 46/50 | 92.00% |
| 最终回答 | 窗口增训草稿，MRV2，图模式，2048／128对齐，窗口分支 4a01608da | 窗口增训草稿，MRV2，图模式，128／128对齐，窗口分支 4a01608da | 34/50 | 68.00% |
| 工具调用 | 窗口增训草稿，MRV2，图模式，2048／128对齐，窗口分支 4a01608da | 窗口增训草稿，MRV2，eager模式，2048／128对齐，窗口分支 4a01608da | 45/50 | 90.00% |
| 最终回答 | 窗口增训草稿，MRV2，图模式，2048／128对齐，窗口分支 4a01608da | 窗口增训草稿，MRV2，eager模式，2048／128对齐，窗口分支 4a01608da | 34/50 | 68.00% |

旧报告中曾用配对精确检验辅助查看一致率变化，工具调用 p=0.2500、最终回答 p=0.4240。该分析不是实验前预设的额外通过门槛，也不能证明数学无损；正文仅保留原始一致题数和结论边界。

## 长上下文完整指标

| 草稿训练方式 | MRV版本／模式 | 推理窗口 | 输入长度档 | 并发 | 题数 | 每题输出token | TTFT ms | TPOT至流结束 ms | 旧可见包口径 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 窗口增训草稿 | MRV2／图模式 | 不开窗 | 24576 | 1 | 4 | 512 | 397.695 | 5.633 | 5.401 |
| 窗口增训草稿 | MRV2／图模式 | 2048／128对齐 | 24576 | 1 | 4 | 512 | 3241.842 | 5.553 | 5.337 |
| 窗口增训草稿 | MRV2／图模式 | 不开窗 | 24576 | 4 | 4 | 512 | 778.865 | 7.609 | 7.333 |
| 窗口增训草稿 | MRV2／图模式 | 2048／128对齐 | 24576 | 4 | 4 | 512 | 889.748 | 7.372 | 7.120 |
| 窗口增训草稿 | MRV2／图模式 | 不开窗 | 29696 | 1 | 4 | 512 | 1863.668 | 5.492 | 5.462 |
| 窗口增训草稿 | MRV2／图模式 | 2048／128对齐 | 29696 | 1 | 4 | 512 | 1975.200 | 5.631 | 5.361 |
| 窗口增训草稿 | MRV2／图模式 | 不开窗 | 29696 | 4 | 4 | 512 | 846.134 | 7.258 | 7.225 |
| 窗口增训草稿 | MRV2／图模式 | 2048／128对齐 | 29696 | 4 | 4 | 512 | 892.160 | 7.176 | 6.837 |

精确输入长度及生成参数：

```json
{
  "requests": [
    {
      "id": "long24576_0",
      "prompt_tokens_template": 24580,
      "requested_target": 24576,
      "source_id": "001"
    },
    {
      "id": "long24576_1",
      "prompt_tokens_template": 24588,
      "requested_target": 24576,
      "source_id": "002"
    },
    {
      "id": "long24576_2",
      "prompt_tokens_template": 24578,
      "requested_target": 24576,
      "source_id": "003"
    },
    {
      "id": "long24576_3",
      "prompt_tokens_template": 24593,
      "requested_target": 24576,
      "source_id": "004"
    },
    {
      "id": "long29696_0",
      "prompt_tokens_template": 29716,
      "requested_target": 29696,
      "source_id": "001"
    },
    {
      "id": "long29696_1",
      "prompt_tokens_template": 29700,
      "requested_target": 29696,
      "source_id": "002"
    },
    {
      "id": "long29696_2",
      "prompt_tokens_template": 29714,
      "requested_target": 29696,
      "source_id": "003"
    },
    {
      "id": "long29696_3",
      "prompt_tokens_template": 29705,
      "requested_target": 29696,
      "source_id": "004"
    }
  ],
  "completion_tokens": 512,
  "temperature": 0,
  "top_p": 1,
  "ignore_eos": true,
  "min_tokens": 512,
  "concurrencies": [
    1,
    4
  ]
}
```

## KV容量与检查结果

0.5个百分点和1个百分点两项预设差值要求来自 [原始验证指令](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3上机验证_prompt.md)。原始请求内容、统计方法和计数保留不变。

| 草稿训练方式 | MRV版本／模式 | 推理窗口 | 推理代码 | KV token容量 | KV blocks |
| --- | --- | --- | --- | --- | --- |
| 不开窗增训草稿 | MRV2／图模式 | 不开窗 | 原代码 fbc3e6b8a | 1,093,632（启动日志） | 未单独采集 |
| 不开窗增训草稿 | MRV2／图模式 | 不开窗 | 窗口分支 4a01608da | 1093632 | 2136 |
| 窗口增训草稿 | MRV2／图模式 | 2048／128对齐 | 窗口分支 4a01608da | 1093632 | 2136 |
| 窗口增训草稿 | MRV2／图模式 | 不开窗 | 窗口分支 4a01608da | 1093632 | 2136 |
| 不开窗增训草稿 | MRV2／图模式 | 2048／128对齐 | 窗口分支 4a01608da | 1093632 | 2136 |
| 窗口增训草稿 | MRV2／图模式 | 128／128对齐 | 窗口分支 4a01608da | 1093632 | 2136 |
| 窗口增训草稿 | MRV2／eager模式 | 2048／128对齐 | 窗口分支 4a01608da | 1093632 | 2136 |

模型使用混合注意力／状态分组，token容量是分组计算后的容量，不能直接用 token容量除以block_size来反推总blocks。原代码基线组未单独保存blocks指标，保留原始token容量，不补造计数。

| 验证事项 | 本次结果 |
| --- | --- |
| 真实权重推理、图捕获和eager执行 | 700/700正式agent请求有效；各组服务日志均有1次预热及100次正式请求成功记录，无未处理异常或OOM |
| 不开窗时更换代码，接受率绝对差≤0.5个百分点 | 工具调用满足；最终回答+0.7032个百分点，未满足 |
| 同一草稿从图模式切到eager，接受率绝对差≤1个百分点 | 工具调用满足；最终回答+1.0904个百分点，未满足 |
| 同一窗口增训草稿关闭窗口，观察截断代价 | 工具调用0.7299个百分点、最终回答0.9526个百分点；原指令未预设硬阈值 |
| 固定2048推理窗口，比较两种训练方式 | 窗口增训草稿接受率更高，分别+6.4147、+5.2728个百分点 |
| 把窗口缩到128，检查结果是否明显受影响 | 最终回答下降20.9824个百分点；工具调用下降6.2570个百分点，幅度小于参考实验预期 |
| 无损输出的工程检查 | 保留96%/66%的代码对照一致率，以及90%/74%的窗口对照一致率；不足以证明所有情况下严格无损 |
| 是否节省KV显存 | 同一窗口增训草稿图模式开／关窗口的token容量及blocks相同 |
| 长上下文是否稳定提速 | 四格单次测量为慢2.53%至快3.11%，未观察到稳定收益 |

旧报告曾附加10个百分点、2个百分点作为描述幅度的辅助界线。它们不是预设判据；此次改写已删除这些额外界线，不用它们硬判通过／不通过。

## 环境、部署与单测记录

使用 `ssh 910b3`，推理容器 `wyd_va_030rc2_1028`，源码路径 `/opt/w00958190/vllm-ascend`。先确认远端仓库干净及editable导入路径；获得授权后核对并停止旧占卡服务。在原代码上完成不开窗草稿基线；后续服务停止后才切换到窗口分支。GitHub fetch超时，使用本地生成的git bundle传输，并核对到完整提交SHA。

执行的单测命令：

```bash
docker exec wyd_va_030rc2_1028 bash -lc 'cd /opt/w00958190/vllm-ascend; ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 pytest -q tests/ut/worker/v2/test_dspark_context_window.py tests/ut/worker/v2/test_dspark_speculator_context_window.py'
```

完整单测输出：

```text
.....................................................                    [100%]
=============================== warnings summary ===============================
../../../usr/local/python3.12.13/lib/python3.12/site-packages/triton/language/extra/cann/extension/math_ops.py:13
  /usr/local/python3.12.13/lib/python3.12/site-packages/triton/language/extra/cann/extension/math_ops.py:13: FutureWarning: triton.language.extra.cann.extension.atan2 is deprecated and will be removed in the next release; use triton.language.extra.cann.libdevice.atan2 instead.
    @_deprecated(replacement="triton.language.extra.cann.libdevice.atan2")

../../../usr/local/python3.12.13/lib/python3.12/site-packages/triton/language/extra/cann/extension/math_ops.py:35
  /usr/local/python3.12.13/lib/python3.12/site-packages/triton/language/extra/cann/extension/math_ops.py:35: FutureWarning: triton.language.extra.cann.extension.isfinited is deprecated and will be removed in the next release; use triton.language.extra.cann.libdevice.isfinited instead.
    @_deprecated(replacement="triton.language.extra.cann.libdevice.isfinited")

../../../usr/local/python3.12.13/lib/python3.12/site-packages/triton/language/extra/cann/extension/math_ops.py:49
  /usr/local/python3.12.13/lib/python3.12/site-packages/triton/language/extra/cann/extension/math_ops.py:49: FutureWarning: triton.language.extra.cann.extension.finitef is deprecated and will be removed in the next release; use triton.language.extra.cann.libdevice.finitef instead.
    @_deprecated(replacement="triton.language.extra.cann.libdevice.finitef")

../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch/jit/_script.py:362: 14 warnings
  /usr/local/python3.12.13/lib/python3.12/site-packages/torch/jit/_script.py:362: DeprecationWarning: `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
53 passed, 17 warnings in 0.52s
```

原件校验、请求数及完成时健康检查：

```json
{
  "actual_commit": "4a01608da7be1950eebc9902e3692d257f304759",
  "repository_clean": true,
  "mrv1_script_unchanged_sha256": "ae44e57c60ec4ee2ab567e37df0761a58e8680ac63292b374216d5ab86794189",
  "source_configs_unchanged": {
    "export_A_agent_ctxwin2048_step50": "d15b306750ecab0aa8b30be86a831eddea7d839fcec849e1e1582f92c56d54c7",
    "export_B_agent_nowindow_step50": "159a58d85bc45971c397ab1b84174c1efd9459ca76ad076b4534bc2d4e8c5f35"
  },
  "weight_symlinks": {
    "A_full": "/opt/w00958190/specforge_tmp/export_A_agent_ctxwin2048_step50/model.safetensors",
    "B_w2048": "/opt/w00958190/specforge_tmp/export_B_agent_nowindow_step50/model.safetensors",
    "A_w128": "/opt/w00958190/specforge_tmp/export_A_agent_ctxwin2048_step50/model.safetensors"
  },
  "final_api_host_pid": 283779,
  "health": {
    "/health": 200,
    "/v1/models": 200
  },
  "service_log_audit": {
    "V0": {
      "successful_post_requests": 101,
      "expected": 101,
      "traceback_present": false,
      "out_of_memory_present": false
    },
    "V1": {
      "successful_post_requests": 101,
      "expected": 101,
      "traceback_present": false,
      "out_of_memory_present": false
    },
    "V2": {
      "successful_post_requests": 101,
      "expected": 101,
      "traceback_present": false,
      "out_of_memory_present": false
    },
    "V3": {
      "successful_post_requests": 101,
      "expected": 101,
      "traceback_present": false,
      "out_of_memory_present": false
    },
    "V4": {
      "successful_post_requests": 101,
      "expected": 101,
      "traceback_present": false,
      "out_of_memory_present": false
    },
    "V5": {
      "successful_post_requests": 101,
      "expected": 101,
      "traceback_present": false,
      "out_of_memory_present": false
    },
    "V6": {
      "successful_post_requests": 101,
      "expected": 101,
      "traceback_present": false,
      "out_of_memory_present": false
    },
    "V3long": {
      "successful_post_requests": 25,
      "expected": 25,
      "traceback_present": false,
      "out_of_memory_present": false
    },
    "V2long": {
      "successful_post_requests": 17,
      "expected": 17,
      "traceback_present": false,
      "out_of_memory_present": false
    }
  },
  "original_benchmark_unchanged_copy_adds_timing_only": true,
  "long_measured_requests": 32
}
```

数据集SHA-256：

```json
{
  "tool_call": "355550b12b335c4cf71514dd0d5d9c1ab43f60730c1202e0be0fe76619dbe565",
  "final_answer": "03d55c07b62d01ee2da3229c31b29e1bd662485924d318b1f1d980d088b0e58b"
}
```

## 异常、位置与处理经过

### 历史工具调用字段在离线模板中渲染失败

位置是长输入长度准备脚本。历史function.arguments为JSON字符串，本地模板需要对象。仅在用于计算长度的消息副本中解析JSON；发往服务的数据保持原样。

```text
[W1030 02:32:37.105225920 FunctionLoader.cpp:48] Warning: LD_PRELOAD detected, FunctionLoader prefers RTLD_DEFAULT for symbol resolution. (function operator())
/usr/local/python3.12.13/lib/python3.12/site-packages/triton/language/extra/cann/extension/math_ops.py:13: FutureWarning: triton.language.extra.cann.extension.atan2 is deprecated and will be removed in the next release; use triton.language.extra.cann.libdevice.atan2 instead.
  @_deprecated(replacement="triton.language.extra.cann.libdevice.atan2")
/usr/local/python3.12.13/lib/python3.12/site-packages/triton/language/extra/cann/extension/math_ops.py:35: FutureWarning: triton.language.extra.cann.extension.isfinited is deprecated and will be removed in the next release; use triton.language.extra.cann.libdevice.isfinited instead.
  @_deprecated(replacement="triton.language.extra.cann.libdevice.isfinited")
/usr/local/python3.12.13/lib/python3.12/site-packages/triton/language/extra/cann/extension/math_ops.py:49: FutureWarning: triton.language.extra.cann.extension.finitef is deprecated and will be removed in the next release; use triton.language.extra.cann.libdevice.finitef instead.
  @_deprecated(replacement="triton.language.extra.cann.libdevice.finitef")
Traceback (most recent call last):
  File "/opt/w00958190/specforge_tmp/logs/mrv2_prepare_long.py", line 21, in <module>
    if length(mid)<target:lo=mid
       ^^^^^^^^^^^
  File "/opt/w00958190/specforge_tmp/logs/mrv2_prepare_long.py", line 17, in length
    return len(tok.apply_chat_template(row['messages'],tools=row.get('tools') or None,tokenize=True,add_generation_prompt=True,enable_thinking=False))
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.12.13/lib/python3.12/site-packages/transformers/tokenization_utils_base.py", line 3121, in apply_chat_template
    rendered_chat, generation_indices = render_jinja_template(
                                        ^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.12.13/lib/python3.12/site-packages/transformers/utils/chat_template_utils.py", line 581, in render_jinja_template
    rendered_chat = compiled_template.render(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.12.13/lib/python3.12/site-packages/jinja2/environment.py", line 1295, in render
    self.environment.handle_exception()
  File "/usr/local/python3.12.13/lib/python3.12/site-packages/jinja2/environment.py", line 942, in handle_exception
    raise rewrite_traceback_stack(source=source)
  File "<template>", line 136, in top-level template code
  File "/usr/local/python3.12.13/lib/python3.12/site-packages/jinja2/filters.py", line 249, in do_items
    raise TypeError("Can only get item pairs from a mapping.")
TypeError: Can only get item pairs from a mapping.
[ERROR] 2026-10-30-02:32:42 (PID:101515, Device:-1, RankID:-1) ERR99999 UNKNOWN applicaiton exception
```

### tokenizer长度读成了返回对象字段数

位置是长输入长度准备脚本。显式设置return_dict=False后，按实际token序列计算长度。

```text
[W1030 02:34:22.457824910 FunctionLoader.cpp:48] Warning: LD_PRELOAD detected, FunctionLoader prefers RTLD_DEFAULT for symbol resolution. (function operator())
/usr/local/python3.12.13/lib/python3.12/site-packages/triton/language/extra/cann/extension/math_ops.py:13: FutureWarning: triton.language.extra.cann.extension.atan2 is deprecated and will be removed in the next release; use triton.language.extra.cann.libdevice.atan2 instead.
  @_deprecated(replacement="triton.language.extra.cann.libdevice.atan2")
/usr/local/python3.12.13/lib/python3.12/site-packages/triton/language/extra/cann/extension/math_ops.py:35: FutureWarning: triton.language.extra.cann.extension.isfinited is deprecated and will be removed in the next release; use triton.language.extra.cann.libdevice.isfinited instead.
  @_deprecated(replacement="triton.language.extra.cann.libdevice.isfinited")
/usr/local/python3.12.13/lib/python3.12/site-packages/triton/language/extra/cann/extension/math_ops.py:49: FutureWarning: triton.language.extra.cann.extension.finitef is deprecated and will be removed in the next release; use triton.language.extra.cann.libdevice.finitef instead.
  @_deprecated(replacement="triton.language.extra.cann.libdevice.finitef")
Traceback (most recent call last):
  File "/opt/w00958190/specforge_tmp/logs/mrv2_prepare_long.py", line 30, in <module>
    assert target<=count<target+100,(target,count)
           ^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: (24576, 2)
[ERROR] 2026-10-30-02:34:27 (PID:103025, Device:-1, RankID:-1) ERR99999 UNKNOWN applicaiton exception
```

### 控制器在后台日志文件创建前立即读取

位置是实验控制器。改为日志出现前等待，从已运行的服务继续，没有因此重启推理服务或重复正式agent评测。

```text
{"round": "V0", "phase": "running", "time": "2026-10-30T10:31:09+0800"}
{"round": "V0", "phase": "complete", "time": "2026-10-30T10:37:29+0800", "commit": "fbc3e6b8ab5d516ae8094b4ff3bf2929987aaf16"}
HEAD is now at 4a01608da fix(spec-decode): tile DSpark graph capture for the window, not the model
{"round": "V1", "phase": "unit_tests", "time": "2026-10-30T10:37:37+0800"}
{"round": "V1", "phase": "running", "time": "2026-10-30T10:38:08+0800"}
Traceback (most recent call last):
  File "/opt/w00958190/specforge_tmp/logs/mrv2_matrix_control.py", line 82, in <module>
    try:main()
        ^^^^^^
  File "/opt/w00958190/specforge_tmp/logs/mrv2_matrix_control.py", line 78, in main
    start(round);wait_round(round)
                 ^^^^^^^^^^^^^^^^^
  File "/opt/w00958190/specforge_tmp/logs/mrv2_matrix_control.py", line 52, in wait_round
    serve=(L/f'serve_{round}.log').read_text(errors='replace')
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib64/python3.11/pathlib.py", line 1058, in read_text
    with self.open(mode='r', encoding=encoding, errors=errors) as f:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib64/python3.11/pathlib.py", line 1044, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/opt/w00958190/specforge_tmp/logs/serve_V1.log'
{"round": "unknown", "phase": "failed", "time": "2026-10-30T10:38:08+0800", "error": "[Errno 2] No such file or directory: '/opt/w00958190/specforge_tmp/logs/serve_V1.log'"}
```

### 缓存重置接口返回404

位置是额外的缓存重置步骤。取消该步骤，未更改服务环境；长组初始前缀缓存状态差异已说明，TTFT仅记录，主比较使用解码TPOT。

```text
{"round": "V3long", "phase": "long_startup", "time": "2026-10-30T11:22:41+0800"}
{"round": "V3long", "phase": "long_benchmark", "time": "2026-10-30T11:27:25+0800", "prompt_tokens": 24576, "concurrency": 1}
{"round": "V3long", "phase": "long_benchmark", "time": "2026-10-30T11:28:34+0800", "prompt_tokens": 24576, "concurrency": 4}
{"round": "V3long", "phase": "long_benchmark", "time": "2026-10-30T11:30:55+0800", "prompt_tokens": 24576, "concurrency": 1}
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/usr/local/python3.12.13/lib/python3.12/site-packages/requests/models.py", line 1167, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 404 Client Error: Not Found for url: http://127.0.0.1:8861/reset_prefix_cache
Traceback (most recent call last):
  File "/opt/w00958190/specforge_tmp/logs/mrv2_long_control.py", line 59, in <module>
    bench('V3long');m.stop('V3long')
    ^^^^^^^^^^^^^^^
  File "/opt/w00958190/specforge_tmp/logs/mrv2_long_control.py", line 24, in bench
    m.dc('python3 -c '+shlex.quote(reset))
  File "/opt/w00958190/specforge_tmp/logs/mrv2_matrix_control.py", line 10, in dc
    return run(args)
           ^^^^^^^^^
  File "/opt/w00958190/specforge_tmp/logs/mrv2_matrix_control.py", line 7, in run
    def run(args,**kw):return subprocess.run(args,check=True,text=True,**kw)
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib64/python3.11/subprocess.py", line 571, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['docker', 'exec', 'wyd_va_030rc2_1028', 'bash', '-lc', 'python3 -c \'import requests;s=requests.Session();s.trust_env=False;r=s.post(\'"\'"\'http://127.0.0.1:8861/reset_prefix_cache\'"\'"\',timeout=30);r.raise_for_status();print(\'"\'"\'PREFIX_RESET\'"\'"\',r.status_code)\'']' returned non-zero exit status 1.
{"round": "long", "phase": "failed", "time": "2026-10-30T11:30:55+0800", "error": "Command '['docker', 'exec', 'wyd_va_030rc2_1028', 'bash', '-lc', 'python3 -c \\'import requests;s=requests.Session();s.trust_env=False;r=s.post(\\'\"\\'\"\\'http://127.0.0.1:8861/reset_prefix_cache\\'\"\\'\"\\',timeout=30);r.raise_for_status();print(\\'\"\\'\"\\'PREFIX_RESET\\'\"\\'\"\\',r.status_code)\\'']' returned non-zero exit status 1."}
Traceback (most recent call last):
  File "/opt/w00958190/specforge_tmp/logs/mrv2_long_control.py", line 59, in <module>
    bench('V3long');m.stop('V3long')
    ^^^^^^^^^^^^^^^
  File "/opt/w00958190/specforge_tmp/logs/mrv2_long_control.py", line 24, in bench
    m.dc('python3 -c '+shlex.quote(reset))
  File "/opt/w00958190/specforge_tmp/logs/mrv2_matrix_control.py", line 10, in dc
    return run(args)
           ^^^^^^^^^
  File "/opt/w00958190/specforge_tmp/logs/mrv2_matrix_control.py", line 7, in run
    def run(args,**kw):return subprocess.run(args,check=True,text=True,**kw)
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib64/python3.11/subprocess.py", line 571, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['docker', 'exec', 'wyd_va_030rc2_1028', 'bash', '-lc', 'python3 -c \'import requests;s=requests.Session();s.trust_env=False;r=s.post(\'"\'"\'http://127.0.0.1:8861/reset_prefix_cache\'"\'"\',timeout=30);r.raise_for_status();print(\'"\'"\'PREFIX_RESET\'"\'"\',r.status_code)\'']' returned non-zero exit status 1.
```

### AOT缓存加载失败后自动重新编译

位置是已有torch AOT编译缓存加载阶段。框架自动重新编译，随后图捕获及真实请求均通过。以下保留全部96行警告和调用栈，完整服务日志另有入口。

```text
(Worker_TP3 pid=159580) WARNING 10-30 03:38:25 [decorators.py:323] Compiling model again due to a load failure from /root/.cache/vllm/torch_compile_cache/torch_aot_compile/7315af84dadb697d04a79071a607b38aca26c399ce55aff10daf6f973e4ce09b/rank_3_0/model, reason: While executing %triton_kernel_wrapper_mutation : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 0, constant_args_idx: 0, grid: [(%floordiv, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X: %reshape_3, Y: %empty_like, W: %contiguous_2, Z: %reshape_4, Rstd: %empty, M: %mul}})
(Worker_TP3 pid=159580) WARNING 10-30 03:38:25 [decorators.py:323] Original traceback:
(Worker_TP3 pid=159580) WARNING 10-30 03:38:25 [decorators.py:323]   File "/vllm-workspace/vllm/vllm/model_executor/models/qwen3_next.py", line 716, in forward
(Worker_TP3 pid=159580) WARNING 10-30 03:38:25 [decorators.py:323]     hidden_states, residual = layer(
(Worker_TP3 pid=159580) WARNING 10-30 03:38:25 [decorators.py:323]   File "/opt/w00958190/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py", line 158, in forward
(Worker_TP3 pid=159580) WARNING 10-30 03:38:25 [decorators.py:323]     hidden_states = self.linear_attn(hidden_states=hidden_states)
(Worker_TP3 pid=159580) WARNING 10-30 03:38:25 [decorators.py:323]   File "/opt/w00958190/vllm-ascend/vllm_ascend/ops/gdn.py", line 444, in forward
(Worker_TP3 pid=159580) WARNING 10-30 03:38:25 [decorators.py:323]     core_attn_out = self.norm(core_attn_out, z)
(Worker_TP3 pid=159580) WARNING 10-30 03:38:25 [decorators.py:323]   File "/vllm-workspace/vllm/vllm/model_executor/custom_op.py", line 136, in forward
(Worker_TP3 pid=159580) WARNING 10-30 03:38:25 [decorators.py:323]     return self._forward_method(*args, **kwargs)
(Worker_TP3 pid=159580) WARNING 10-30 03:38:25 [decorators.py:323]   File "/opt/w00958190/vllm-ascend/vllm_ascend/ops/layernorm.py", line 213, in forward_oot
(Worker_TP3 pid=159580) WARNING 10-30 03:38:25 [decorators.py:323]     return LayerNormFn.apply(x, self.weight, self.bias, z, self.eps, self.group_size, self.norm_before_gate, True)
(Worker_TP3 pid=159580) WARNING 10-30 03:38:25 [decorators.py:323]   File "/opt/w00958190/vllm-ascend/vllm_ascend/ops/layernorm.py", line 155, in forward
(Worker_TP3 pid=159580) WARNING 10-30 03:38:25 [decorators.py:323]     y, mean, rstd = layer_norm_fwd_npu(
(Worker_TP3 pid=159580) WARNING 10-30 03:38:25 [decorators.py:323]   File "/opt/w00958190/vllm-ascend/vllm_ascend/ops/triton/layernorm_gated.py", line 148, in layer_norm_fwd_npu
(Worker_TP3 pid=159580) WARNING 10-30 03:38:25 [decorators.py:323]     _layer_norm_fwd_1pass_kernel_npu[grid](
(Worker_TP3 pid=159580) WARNING 10-30 03:38:25 [decorators.py:323]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/triton/runtime/jit.py", line 370, in <lambda>
(Worker_TP3 pid=159580) WARNING 10-30 03:38:25 [decorators.py:323]     return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
(Worker_TP3 pid=159580) WARNING 10-30 03:38:25 [decorators.py:323]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/triton/runtime/autotuner.py", line 459, in run
(Worker_TP3 pid=159580) WARNING 10-30 03:38:25 [decorators.py:323]     return self.fn.run(*args, **kwargs)
(Worker_TP3 pid=159580) WARNING 10-30 03:38:25 [decorators.py:323]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/triton/runtime/autotuner.py", line 459, in run
(Worker_TP3 pid=159580) WARNING 10-30 03:38:25 [decorators.py:323]     return self.fn.run(*args, **kwargs)
(Worker_TP3 pid=159580) WARNING 10-30 03:38:25 [decorators.py:323]
(Worker_TP3 pid=159580) WARNING 10-30 03:38:25 [decorators.py:323] Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs)
(Worker_TP0 pid=159577) WARNING 10-30 03:38:25 [decorators.py:323] Compiling model again due to a load failure from /root/.cache/vllm/torch_compile_cache/torch_aot_compile/7315af84dadb697d04a79071a607b38aca26c399ce55aff10daf6f973e4ce09b/rank_0_0/model, reason: While executing %triton_kernel_wrapper_mutation : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 0, constant_args_idx: 0, grid: [(%floordiv, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X: %reshape_3, Y: %empty_like, W: %contiguous_2, Z: %reshape_4, Rstd: %empty, M: %mul}})
(Worker_TP0 pid=159577) WARNING 10-30 03:38:25 [decorators.py:323] Original traceback:
(Worker_TP0 pid=159577) WARNING 10-30 03:38:25 [decorators.py:323]   File "/vllm-workspace/vllm/vllm/model_executor/models/qwen3_next.py", line 716, in forward
(Worker_TP0 pid=159577) WARNING 10-30 03:38:25 [decorators.py:323]     hidden_states, residual = layer(
(Worker_TP0 pid=159577) WARNING 10-30 03:38:25 [decorators.py:323]   File "/opt/w00958190/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py", line 158, in forward
(Worker_TP0 pid=159577) WARNING 10-30 03:38:25 [decorators.py:323]     hidden_states = self.linear_attn(hidden_states=hidden_states)
(Worker_TP0 pid=159577) WARNING 10-30 03:38:25 [decorators.py:323]   File "/opt/w00958190/vllm-ascend/vllm_ascend/ops/gdn.py", line 444, in forward
(Worker_TP0 pid=159577) WARNING 10-30 03:38:25 [decorators.py:323]     core_attn_out = self.norm(core_attn_out, z)
(Worker_TP0 pid=159577) WARNING 10-30 03:38:25 [decorators.py:323]   File "/vllm-workspace/vllm/vllm/model_executor/custom_op.py", line 136, in forward
(Worker_TP0 pid=159577) WARNING 10-30 03:38:25 [decorators.py:323]     return self._forward_method(*args, **kwargs)
(Worker_TP0 pid=159577) WARNING 10-30 03:38:25 [decorators.py:323]   File "/opt/w00958190/vllm-ascend/vllm_ascend/ops/layernorm.py", line 213, in forward_oot
(Worker_TP0 pid=159577) WARNING 10-30 03:38:25 [decorators.py:323]     return LayerNormFn.apply(x, self.weight, self.bias, z, self.eps, self.group_size, self.norm_before_gate, True)
(Worker_TP0 pid=159577) WARNING 10-30 03:38:25 [decorators.py:323]   File "/opt/w00958190/vllm-ascend/vllm_ascend/ops/layernorm.py", line 155, in forward
(Worker_TP0 pid=159577) WARNING 10-30 03:38:25 [decorators.py:323]     y, mean, rstd = layer_norm_fwd_npu(
(Worker_TP0 pid=159577) WARNING 10-30 03:38:25 [decorators.py:323]   File "/opt/w00958190/vllm-ascend/vllm_ascend/ops/triton/layernorm_gated.py", line 148, in layer_norm_fwd_npu
(Worker_TP0 pid=159577) WARNING 10-30 03:38:25 [decorators.py:323]     _layer_norm_fwd_1pass_kernel_npu[grid](
(Worker_TP0 pid=159577) WARNING 10-30 03:38:25 [decorators.py:323]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/triton/runtime/jit.py", line 370, in <lambda>
(Worker_TP0 pid=159577) WARNING 10-30 03:38:25 [decorators.py:323]     return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
(Worker_TP0 pid=159577) WARNING 10-30 03:38:25 [decorators.py:323]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/triton/runtime/autotuner.py", line 459, in run
(Worker_TP0 pid=159577) WARNING 10-30 03:38:25 [decorators.py:323]     return self.fn.run(*args, **kwargs)
(Worker_TP0 pid=159577) WARNING 10-30 03:38:25 [decorators.py:323]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/triton/runtime/autotuner.py", line 459, in run
(Worker_TP0 pid=159577) WARNING 10-30 03:38:25 [decorators.py:323]     return self.fn.run(*args, **kwargs)
(Worker_TP0 pid=159577) WARNING 10-30 03:38:25 [decorators.py:323]
(Worker_TP0 pid=159577) WARNING 10-30 03:38:25 [decorators.py:323] Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs)
(Worker_TP1 pid=159578) WARNING 10-30 03:38:25 [decorators.py:323] Compiling model again due to a load failure from /root/.cache/vllm/torch_compile_cache/torch_aot_compile/7315af84dadb697d04a79071a607b38aca26c399ce55aff10daf6f973e4ce09b/rank_1_0/model, reason: While executing %triton_kernel_wrapper_mutation : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 0, constant_args_idx: 0, grid: [(%floordiv, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X: %reshape_3, Y: %empty_like, W: %contiguous_2, Z: %reshape_4, Rstd: %empty, M: %mul}})
(Worker_TP1 pid=159578) WARNING 10-30 03:38:25 [decorators.py:323] Original traceback:
(Worker_TP1 pid=159578) WARNING 10-30 03:38:25 [decorators.py:323]   File "/vllm-workspace/vllm/vllm/model_executor/models/qwen3_next.py", line 716, in forward
(Worker_TP1 pid=159578) WARNING 10-30 03:38:25 [decorators.py:323]     hidden_states, residual = layer(
(Worker_TP1 pid=159578) WARNING 10-30 03:38:25 [decorators.py:323]   File "/opt/w00958190/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py", line 158, in forward
(Worker_TP1 pid=159578) WARNING 10-30 03:38:25 [decorators.py:323]     hidden_states = self.linear_attn(hidden_states=hidden_states)
(Worker_TP1 pid=159578) WARNING 10-30 03:38:25 [decorators.py:323]   File "/opt/w00958190/vllm-ascend/vllm_ascend/ops/gdn.py", line 444, in forward
(Worker_TP1 pid=159578) WARNING 10-30 03:38:25 [decorators.py:323]     core_attn_out = self.norm(core_attn_out, z)
(Worker_TP1 pid=159578) WARNING 10-30 03:38:25 [decorators.py:323]   File "/vllm-workspace/vllm/vllm/model_executor/custom_op.py", line 136, in forward
(Worker_TP1 pid=159578) WARNING 10-30 03:38:25 [decorators.py:323]     return self._forward_method(*args, **kwargs)
(Worker_TP1 pid=159578) WARNING 10-30 03:38:25 [decorators.py:323]   File "/opt/w00958190/vllm-ascend/vllm_ascend/ops/layernorm.py", line 213, in forward_oot
(Worker_TP1 pid=159578) WARNING 10-30 03:38:25 [decorators.py:323]     return LayerNormFn.apply(x, self.weight, self.bias, z, self.eps, self.group_size, self.norm_before_gate, True)
(Worker_TP1 pid=159578) WARNING 10-30 03:38:25 [decorators.py:323]   File "/opt/w00958190/vllm-ascend/vllm_ascend/ops/layernorm.py", line 155, in forward
(Worker_TP1 pid=159578) WARNING 10-30 03:38:25 [decorators.py:323]     y, mean, rstd = layer_norm_fwd_npu(
(Worker_TP1 pid=159578) WARNING 10-30 03:38:25 [decorators.py:323]   File "/opt/w00958190/vllm-ascend/vllm_ascend/ops/triton/layernorm_gated.py", line 148, in layer_norm_fwd_npu
(Worker_TP1 pid=159578) WARNING 10-30 03:38:25 [decorators.py:323]     _layer_norm_fwd_1pass_kernel_npu[grid](
(Worker_TP1 pid=159578) WARNING 10-30 03:38:25 [decorators.py:323]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/triton/runtime/jit.py", line 370, in <lambda>
(Worker_TP1 pid=159578) WARNING 10-30 03:38:25 [decorators.py:323]     return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
(Worker_TP1 pid=159578) WARNING 10-30 03:38:25 [decorators.py:323]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/triton/runtime/autotuner.py", line 459, in run
(Worker_TP1 pid=159578) WARNING 10-30 03:38:25 [decorators.py:323]     return self.fn.run(*args, **kwargs)
(Worker_TP1 pid=159578) WARNING 10-30 03:38:25 [decorators.py:323]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/triton/runtime/autotuner.py", line 459, in run
(Worker_TP1 pid=159578) WARNING 10-30 03:38:25 [decorators.py:323]     return self.fn.run(*args, **kwargs)
(Worker_TP1 pid=159578) WARNING 10-30 03:38:25 [decorators.py:323]
(Worker_TP1 pid=159578) WARNING 10-30 03:38:25 [decorators.py:323] Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs)
(Worker_TP2 pid=159579) WARNING 10-30 03:38:25 [decorators.py:323] Compiling model again due to a load failure from /root/.cache/vllm/torch_compile_cache/torch_aot_compile/7315af84dadb697d04a79071a607b38aca26c399ce55aff10daf6f973e4ce09b/rank_2_0/model, reason: While executing %triton_kernel_wrapper_mutation : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 0, constant_args_idx: 0, grid: [(%floordiv, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X: %reshape_3, Y: %empty_like, W: %contiguous_2, Z: %reshape_4, Rstd: %empty, M: %mul}})
(Worker_TP2 pid=159579) WARNING 10-30 03:38:25 [decorators.py:323] Original traceback:
(Worker_TP2 pid=159579) WARNING 10-30 03:38:25 [decorators.py:323]   File "/vllm-workspace/vllm/vllm/model_executor/models/qwen3_next.py", line 716, in forward
(Worker_TP2 pid=159579) WARNING 10-30 03:38:25 [decorators.py:323]     hidden_states, residual = layer(
(Worker_TP2 pid=159579) WARNING 10-30 03:38:25 [decorators.py:323]   File "/opt/w00958190/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py", line 158, in forward
(Worker_TP2 pid=159579) WARNING 10-30 03:38:25 [decorators.py:323]     hidden_states = self.linear_attn(hidden_states=hidden_states)
(Worker_TP2 pid=159579) WARNING 10-30 03:38:25 [decorators.py:323]   File "/opt/w00958190/vllm-ascend/vllm_ascend/ops/gdn.py", line 444, in forward
(Worker_TP2 pid=159579) WARNING 10-30 03:38:25 [decorators.py:323]     core_attn_out = self.norm(core_attn_out, z)
(Worker_TP2 pid=159579) WARNING 10-30 03:38:25 [decorators.py:323]   File "/vllm-workspace/vllm/vllm/model_executor/custom_op.py", line 136, in forward
(Worker_TP2 pid=159579) WARNING 10-30 03:38:25 [decorators.py:323]     return self._forward_method(*args, **kwargs)
(Worker_TP2 pid=159579) WARNING 10-30 03:38:25 [decorators.py:323]   File "/opt/w00958190/vllm-ascend/vllm_ascend/ops/layernorm.py", line 213, in forward_oot
(Worker_TP2 pid=159579) WARNING 10-30 03:38:25 [decorators.py:323]     return LayerNormFn.apply(x, self.weight, self.bias, z, self.eps, self.group_size, self.norm_before_gate, True)
(Worker_TP2 pid=159579) WARNING 10-30 03:38:25 [decorators.py:323]   File "/opt/w00958190/vllm-ascend/vllm_ascend/ops/layernorm.py", line 155, in forward
(Worker_TP2 pid=159579) WARNING 10-30 03:38:25 [decorators.py:323]     y, mean, rstd = layer_norm_fwd_npu(
(Worker_TP2 pid=159579) WARNING 10-30 03:38:25 [decorators.py:323]   File "/opt/w00958190/vllm-ascend/vllm_ascend/ops/triton/layernorm_gated.py", line 148, in layer_norm_fwd_npu
(Worker_TP2 pid=159579) WARNING 10-30 03:38:25 [decorators.py:323]     _layer_norm_fwd_1pass_kernel_npu[grid](
(Worker_TP2 pid=159579) WARNING 10-30 03:38:25 [decorators.py:323]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/triton/runtime/jit.py", line 370, in <lambda>
(Worker_TP2 pid=159579) WARNING 10-30 03:38:25 [decorators.py:323]     return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
(Worker_TP2 pid=159579) WARNING 10-30 03:38:25 [decorators.py:323]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/triton/runtime/autotuner.py", line 459, in run
(Worker_TP2 pid=159579) WARNING 10-30 03:38:25 [decorators.py:323]     return self.fn.run(*args, **kwargs)
(Worker_TP2 pid=159579) WARNING 10-30 03:38:25 [decorators.py:323]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/triton/runtime/autotuner.py", line 459, in run
(Worker_TP2 pid=159579) WARNING 10-30 03:38:25 [decorators.py:323]     return self.fn.run(*args, **kwargs)
(Worker_TP2 pid=159579) WARNING 10-30 03:38:25 [decorators.py:323]
(Worker_TP2 pid=159579) WARNING 10-30 03:38:25 [decorators.py:323] Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs)
```

准备阶段服务器缺少rg，完整错误为 `bash: line 1: rg: command not found`，随后改用grep，没有安装工具。一次本地路径检查误用了dspark.py，完整错误为 `sed: vllm_ascend/worker/v2/spec_decode/dspark.py: No such file or directory`，后按文件列表改读dspark/speculator.py。GitHub fetch超时退出124，没有额外stderr。

## 完整启动脚本与各配置启动命令

以下为本次实验实际使用的历史脚本和命令。保留原文用于复现本次实验，不自动重放；后续运行遵循最新的服务启动和日志规范。

### MRV2实验启动脚本原文

```bash
#!/bin/bash
rm -rf .torchair_cache/

# ================= 1. 参数解析与默认值配置 =================

# 设置默认值
MODE="compile"
USE_ASYNC="true"
QUANT="false"
USE_LORA="false"     # (新增) 默认不开启 LoRA
SPEC_STEPS=0
MAX_SEQS=2
PORT=8841
GPU_GROUP="last4"
PROFILER_DIR=""     # 默认不开启 Profiler，如果传入路径则开启
SPEC_METHOD="dspark" # (新增) 默认投机解码方法
TP_SIZE=4            # (新增) 默认 Tensor Parallel Size 为 4
MAX_MODEL_LEN=200000
GPU_MEM_UTIL=0.9

# 打印帮助信息的函数
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --mode <string>        运行模式: eager 或 compile (默认 compile)"
    echo "  --async <bool>         是否开启异步调度: true 或 false (默认 true)"
    echo "  --quant <bool>         是否使用 w8a8 量化模型: true 或 false (默认 false)"
    echo "  --lora <bool>          是否挂载 LoRA 模块: true 或 false (默认 false)"
    echo "  --tp <int>             Tensor Parallel Size (默认 4。若传 8，则自动使用 0-7 所有卡，无需指定 --gpus)" # (新增)
    echo "  --spec-steps <int>     投机步数 num_speculative_tokens (默认 0)"
    echo "  --spec-method <string> 投机解码方法: dspark 或 mtp (默认 dspark)"
    echo "  --max-seqs <int>       最大并发 max_num_seqs (默认 2)"
    echo "  --max-model-len <int>  最大上下文长度 max_model_len (默认 200000)"
    echo "  --gpu-mem-util <float> gpu_memory_utilization (默认 0.9)"
    echo "  --port <int>           服务启用的端口号 (默认 8841)"
    echo "  --gpus <string>        使用的 NPU 显卡组: first4 (0-3) 或 last4 (4-7) (默认 last4)"
    echo "  --profiler-dir <dir>   开启 Profiling 并设置结果输出目录 (默认不开启)"
    echo "  --help                 显示此帮助信息"
    echo ""
    echo "Example:"
    echo "  bash $0 --mode compile --lora true --port 8080 --tp 8"
    exit 1
}

# 核心解析循环
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift 2 ;;
        --async) USE_ASYNC="$2"; shift 2 ;;
        --quant) QUANT="$2"; shift 2 ;;
        --lora) USE_LORA="$2"; shift 2 ;;
        --tp) TP_SIZE="$2"; shift 2 ;; # (新增) 解析 tp 参数
        --spec-steps) SPEC_STEPS="$2"; shift 2 ;;
        --spec-method) SPEC_METHOD="$2"; shift 2 ;;
        --max-seqs) MAX_SEQS="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpu-mem-util) GPU_MEM_UTIL="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --gpus) GPU_GROUP="$2"; shift 2 ;;
        --profiler-dir) PROFILER_DIR="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "❌ 未知参数: $1"; usage ;;
    esac
done

# ================= 2. 环境变量配置 =================

HOOK_DIR=/opt/z00830407/hook/
if [ -d "$HOOK_DIR" ]; then
    export PYTHONPATH=$HOOK_DIR:$PYTHONPATH
fi

# 如果传入了 profiler 路径，则设置对应环境变量
if [ -n "$PROFILER_DIR" ]; then
    export VLLM_TORCH_PROFILER_DIR="$PROFILER_DIR"
fi

export model_hook=0
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_IF_BASE_PORT=60003
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_CONNECT_TIMEOUT=7200


# 030rc2 镜像没有 omni vendor；FIA_SINK=2 (FA4) 只需要 flash_attn_npu_4
OMNI_ENV=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/omni_custom_transformer/bin/set_env.bash
if [ -f "$OMNI_ENV" ]; then
    source "$OMNI_ENV"
else
    echo ">>> [Env] $OMNI_ENV not found, skipped."
fi

# Debugging envs
# export VLLM_LOGGING_LEVEL=DEBUG
# export ASCEND_LAUNCH_BLOCKING=1

# (修改) 根据 --tp 和 --gpus 动态分配物理卡号
if [ "$TP_SIZE" -eq 8 ]; then
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    echo ">>> [GPUs] TP=8, Allocated All 8 NPUs: 0, 1, 2, 3, 4, 5, 6, 7"
else
    if [ "$GPU_GROUP" == "first4" ]; then
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
        echo ">>> [GPUs] Allocated First 4 NPUs: 0, 1, 2, 3"
    elif [ "$GPU_GROUP" == "last4" ]; then
        export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
        echo ">>> [GPUs] Allocated Last 4 NPUs: 4, 5, 6, 7"
    else
        echo "❌ Error: Invalid gpu group '$GPU_GROUP'. Must be 'first4' or 'last4'."
        exit 1
    fi
fi

# ================= 3. 参数逻辑构建 =================

# 根据 --quant 决定模型路径与量化参数
if [ "$QUANT" == "true" ]; then
    echo ">>> [Quantization] ENABLED (w8a8)"
    MODEL_PATH="/opt/models/Qwen3.6-27B-w8a8"
    QUANT_ARGS=(--quantization "ascend")
else
    echo ">>> [Quantization] DISABLED (fp16/bf16)"
    MODEL_PATH="/opt/models/Qwen3.8-27B"
    QUANT_ARGS=()
fi

# 根据 --spec-method 动态组装 speculative-config
if [ "$SPEC_METHOD" == "dspark" ]; then
    echo ">>> [Speculative] Method: dspark"
    # MRV2 reads the draft context window from its checkpoint config.
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK=2
    export VLLM_ASCEND_KV_GROUP_MIN_SIZE=16
    DRAFT_MODEL=${DRAFT_MODEL:-/opt/models/Qwen3.8-27B-DSpark-Base}
    SPEC_CONFIG='{"num_speculative_tokens": '"$SPEC_STEPS"', "method":"dspark", "model": "'"$DRAFT_MODEL"'"}'
    # "enforce_eager": true
elif [ "$SPEC_METHOD" == "mtp" ]; then
    echo ">>> [Speculative] Method: mtp"
    export VLLM_USE_V2_MODEL_RUNNER=1
    SPEC_CONFIG='{"method": "mtp", "num_speculative_tokens": '"$SPEC_STEPS"'}'
else
    echo "❌ Error: Invalid spec method '$SPEC_METHOD'. Must be 'dspark' or 'mtp'."
    exit 1
fi

# (修改) 定义基础参数数组，tensor-parallel-size 改为读取变量 $TP_SIZE
args=(
    --model "$MODEL_PATH"
    --served-model-name "qwen3.8-27b-bare"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_SEQS"
    --data-parallel-size 1
    --tensor-parallel-size "$TP_SIZE"
    --port "$PORT"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --trust-remote-code
    --tokenizer-mode auto
    --tool-call-parser qwen3_coder
    --enable-auto-tool-choice
    --reasoning-parser qwen3
    --allowed-local-media-path /
    --enable-request-id-headers
    --speculative-config "$SPEC_CONFIG"
    --default-chat-template-kwargs '{"enable_thinking": false}'
    --enable-prefix-caching
)

echo ">>> [Draft] $DRAFT_MODEL"

# 附加量化参数
if [ ${#QUANT_ARGS[@]} -gt 0 ]; then
    args+=("${QUANT_ARGS[@]}")
fi

# 处理 LoRA 附加参数
if [ "$USE_LORA" == "true" ]; then
    echo ">>> [LoRA] ENABLED."
    args+=(
        --enable-lora
        --fully-sharded-loras
        --max-lora-rank 8
        --lora-modules '{"name": "test-lora", "path": "/opt/foundation_model/Qwen3.6-27B-NAIE-Lora", "base_model_name": "qwen3.8-27b-bare"}'
    )
else
    echo ">>> [LoRA] DISABLED."
fi

# 处理 Profiler
if [ -n "$PROFILER_DIR" ]; then
    echo ">>> [Profiler] ENABLED. Output dir: $PROFILER_DIR"
    # 使用 JSON 字符串拼接，将路径变量注入
    PROFILER_CONFIG='{"profiler": "torch", "torch_profiler_dir": "'"$PROFILER_DIR"'", "torch_profiler_with_stack": true}'
    args+=(--profiler-config "$PROFILER_CONFIG")
else
    echo ">>> [Profiler] DISABLED."
fi

# 处理 Mode 与 图编译参数自动计算
if [ "$MODE" == "eager" ]; then
    echo ">>> [Mode] Running in EAGER mode..."
    args+=(--enforce-eager)
elif [ "$MODE" == "compile" ]; then
    echo ">>> [Mode] Running in GRAPH COMPILATION mode..."

    STEP_SIZE=$(( 1 + SPEC_STEPS ))
    MAX_CAPTURE=$(( MAX_SEQS * STEP_SIZE ))
    CAPTURE_SIZES_STR=$(seq -s ', ' $STEP_SIZE $STEP_SIZE $MAX_CAPTURE)

    echo ">>> [Compile] Generated cudagraph_capture_sizes: [$CAPTURE_SIZES_STR]"

    COMPILE_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":['"$CAPTURE_SIZES_STR"']}'

    args+=(--compilation-config "$COMPILE_CONFIG")
else
    echo "❌ Error: Invalid mode '$MODE'."
    usage
fi

# 处理 Async
if [ "$USE_ASYNC" == "true" ]; then
    echo ">>> [Async] Async Engine Execution ENABLED."
    args+=(--async_scheduling)
else
    echo ">>> [Async] Async Engine Execution DISABLED."
fi

echo "---------------------------------------------------"
echo "Final Executing Command:"
echo "python -m vllm.entrypoints.openai.api_server ${args[*]}"
echo "---------------------------------------------------"


echo $VLLM_USE_V2_MODEL_RUNNER
# ================= 4. 启动服务 =================

python -m vllm.entrypoints.openai.api_server "${args[@]}"
```

### 正式agent评测：不开窗增训草稿，MRV2，图模式，不开窗，原代码 fbc3e6b8a

```bash
cd /opt/w00958190; export DRAFT_MODEL=/opt/w00958190/specforge_tmp/export_B_agent_nowindow_step50; unset DRAFT_WINDOW; bash /opt/w00958190/specforge_tmp/run_server_qwen3.8_dspark_mrv2_ctxwin.sh --mode compile --spec-method dspark --spec-steps 7 --max-seqs 8 --tp 4 --gpus last4 --gpu-mem-util 0.9 --max-model-len 32768 --port 8861
```

### 正式agent评测：不开窗增训草稿，MRV2，图模式，不开窗，窗口分支 4a01608da

```bash
cd /opt/w00958190; export DRAFT_MODEL=/opt/w00958190/specforge_tmp/export_B_agent_nowindow_step50; unset DRAFT_WINDOW; bash /opt/w00958190/specforge_tmp/run_server_qwen3.8_dspark_mrv2_ctxwin.sh --mode compile --spec-method dspark --spec-steps 7 --max-seqs 8 --tp 4 --gpus last4 --gpu-mem-util 0.9 --max-model-len 32768 --port 8861
```

### 正式agent评测：窗口增训草稿，MRV2，图模式，2048／128对齐，窗口分支 4a01608da

```bash
cd /opt/w00958190; export DRAFT_MODEL=/opt/w00958190/specforge_tmp/export_A_agent_ctxwin2048_step50; unset DRAFT_WINDOW; bash /opt/w00958190/specforge_tmp/run_server_qwen3.8_dspark_mrv2_ctxwin.sh --mode compile --spec-method dspark --spec-steps 7 --max-seqs 8 --tp 4 --gpus last4 --gpu-mem-util 0.9 --max-model-len 32768 --port 8861
```

### 正式agent评测：窗口增训草稿，MRV2，图模式，不开窗，窗口分支 4a01608da

```bash
cd /opt/w00958190; export DRAFT_MODEL=/opt/w00958190/specforge_tmp/ctxwin_variants/A_full; unset DRAFT_WINDOW; bash /opt/w00958190/specforge_tmp/run_server_qwen3.8_dspark_mrv2_ctxwin.sh --mode compile --spec-method dspark --spec-steps 7 --max-seqs 8 --tp 4 --gpus last4 --gpu-mem-util 0.9 --max-model-len 32768 --port 8861
```

### 正式agent评测：不开窗增训草稿，MRV2，图模式，2048／128对齐，窗口分支 4a01608da

```bash
cd /opt/w00958190; export DRAFT_MODEL=/opt/w00958190/specforge_tmp/ctxwin_variants/B_w2048; unset DRAFT_WINDOW; bash /opt/w00958190/specforge_tmp/run_server_qwen3.8_dspark_mrv2_ctxwin.sh --mode compile --spec-method dspark --spec-steps 7 --max-seqs 8 --tp 4 --gpus last4 --gpu-mem-util 0.9 --max-model-len 32768 --port 8861
```

### 正式agent评测：窗口增训草稿，MRV2，图模式，128／128对齐，窗口分支 4a01608da

```bash
cd /opt/w00958190; export DRAFT_MODEL=/opt/w00958190/specforge_tmp/ctxwin_variants/A_w128; unset DRAFT_WINDOW; bash /opt/w00958190/specforge_tmp/run_server_qwen3.8_dspark_mrv2_ctxwin.sh --mode compile --spec-method dspark --spec-steps 7 --max-seqs 8 --tp 4 --gpus last4 --gpu-mem-util 0.9 --max-model-len 32768 --port 8861
```

### 正式agent评测：窗口增训草稿，MRV2，eager模式，2048／128对齐，窗口分支 4a01608da

```bash
cd /opt/w00958190; export DRAFT_MODEL=/opt/w00958190/specforge_tmp/export_A_agent_ctxwin2048_step50; unset DRAFT_WINDOW; bash /opt/w00958190/specforge_tmp/run_server_qwen3.8_dspark_mrv2_ctxwin.sh --mode eager --spec-method dspark --spec-steps 7 --max-seqs 8 --tp 4 --gpus last4 --gpu-mem-util 0.9 --max-model-len 32768 --port 8861
```

### 长上下文速度组：窗口增训草稿，MRV2，图模式，不开窗，窗口分支 4a01608da

```bash
cd /opt/w00958190; export DRAFT_MODEL=/opt/w00958190/specforge_tmp/ctxwin_variants/A_full; unset DRAFT_WINDOW; bash /opt/w00958190/specforge_tmp/run_server_qwen3.8_dspark_mrv2_ctxwin.sh --mode compile --spec-method dspark --spec-steps 7 --max-seqs 8 --tp 4 --gpus last4 --gpu-mem-util 0.9 --max-model-len 32768 --port 8861
```

### 长上下文速度组：窗口增训草稿，MRV2，图模式，2048／128对齐，窗口分支 4a01608da

```bash
cd /opt/w00958190; export DRAFT_MODEL=/opt/w00958190/specforge_tmp/export_A_agent_ctxwin2048_step50; unset DRAFT_WINDOW; bash /opt/w00958190/specforge_tmp/run_server_qwen3.8_dspark_mrv2_ctxwin.sh --mode compile --spec-method dspark --spec-steps 7 --max-seqs 8 --tp 4 --gpus last4 --gpu-mem-util 0.9 --max-model-len 32768 --port 8861
```

## 原始结果、完整日志及逐题输出入口

本地证据目录：/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据。服务器原始证据目录：`/opt/w00958190/specforge_tmp/logs/`。以下链接指向本地完整文件；服务器有相应同名记录。

[接受率、原始计数、逐题不同文本及启动证据汇总](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/mrv2_summary.json)

### 不开窗增训草稿，MRV2，图模式，不开窗，原代码 fbc3e6b8a（正式agent评测）

[服务完整日志](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/serve_V0.log)；[预热与真实请求探测日志](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/probe_final_answer_V0.log)

- [工具调用：完整请求和结果](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/bench_V0_tool_call_2026-10-30_02-35-35.jsonl)
- [最终回答：完整请求和结果](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/bench_V0_final_answer_2026-10-30_02-36-29.jsonl)

### 不开窗增训草稿，MRV2，图模式，不开窗，窗口分支 4a01608da（正式agent评测）

[服务完整日志](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/serve_V1.log)；[预热与真实请求探测日志](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/probe_final_answer_V1.log)

- [工具调用：完整请求和结果](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/bench_V1_tool_call_2026-10-30_02-43-36.jsonl)
- [最终回答：完整请求和结果](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/bench_V1_final_answer_2026-10-30_02-44-19.jsonl)

### 窗口增训草稿，MRV2，图模式，2048／128对齐，窗口分支 4a01608da（正式agent评测）

[服务完整日志](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/serve_V2.log)；[预热与真实请求探测日志](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/probe_final_answer_V2.log)

- [工具调用：完整请求和结果](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/bench_V2_tool_call_2026-10-30_02-50-11.jsonl)
- [最终回答：完整请求和结果](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/bench_V2_final_answer_2026-10-30_02-50-50.jsonl)

### 窗口增训草稿，MRV2，图模式，不开窗，窗口分支 4a01608da（正式agent评测）

[服务完整日志](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/serve_V3.log)；[预热与真实请求探测日志](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/probe_final_answer_V3.log)

- [工具调用：完整请求和结果](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/bench_V3_tool_call_2026-10-30_02-56-51.jsonl)
- [最终回答：完整请求和结果](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/bench_V3_final_answer_2026-10-30_02-57-31.jsonl)

### 不开窗增训草稿，MRV2，图模式，2048／128对齐，窗口分支 4a01608da（正式agent评测）

[服务完整日志](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/serve_V4.log)；[预热与真实请求探测日志](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/probe_final_answer_V4.log)

- [工具调用：完整请求和结果](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/bench_V4_tool_call_2026-10-30_03-06-06.jsonl)
- [最终回答：完整请求和结果](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/bench_V4_final_answer_2026-10-30_03-06-47.jsonl)

### 窗口增训草稿，MRV2，图模式，128／128对齐，窗口分支 4a01608da（正式agent评测）

[服务完整日志](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/serve_V5.log)；[预热与真实请求探测日志](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/probe_final_answer_V5.log)

- [工具调用：完整请求和结果](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/bench_V5_tool_call_2026-10-30_03-13-12.jsonl)
- [最终回答：完整请求和结果](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/bench_V5_final_answer_2026-10-30_03-13-52.jsonl)

### 窗口增训草稿，MRV2，eager模式，2048／128对齐，窗口分支 4a01608da（正式agent评测）

[服务完整日志](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/serve_V6.log)；[预热与真实请求探测日志](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/probe_final_answer_V6.log)

- [工具调用：完整请求和结果](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/bench_V6_tool_call_2026-10-30_03-19-11.jsonl)
- [最终回答：完整请求和结果](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/bench_V6_final_answer_2026-10-30_03-20-15.jsonl)

### 窗口增训草稿，MRV2，图模式，不开窗，窗口分支 4a01608da（长上下文）

[服务完整日志](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/serve_V3long.log)；[预热与真实请求探测日志](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/probe_final_answer_V3long.log)

- [24576长度档、并发1：完整请求和结果](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/bench_V3long_24576_c1_2026-10-30_03-34-19.jsonl)
- [24576长度档、并发4：完整请求和结果](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/bench_V3long_24576_c4_2026-10-30_03-34-45.jsonl)
- [29696长度档、并发1：完整请求和结果](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/bench_V3long_29696_c1_2026-10-30_03-35-03.jsonl)
- [29696长度档、并发4：完整请求和结果](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/bench_V3long_29696_c4_2026-10-30_03-35-34.jsonl)

### 窗口增训草稿，MRV2，图模式，2048／128对齐，窗口分支 4a01608da（长上下文）

[服务完整日志](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/serve_V2long.log)；[预热与真实请求探测日志](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/probe_final_answer_V2long.log)

- [24576长度档、并发1：完整请求和结果](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/bench_V2long_24576_c1_2026-10-30_03-40-45.jsonl)
- [24576长度档、并发4：完整请求和结果](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/bench_V2long_24576_c4_2026-10-30_03-41-22.jsonl)
- [29696长度档、并发1：完整请求和结果](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/bench_V2long_29696_c1_2026-10-30_03-41-39.jsonl)
- [29696长度档、并发4：完整请求和结果](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/bench_V2long_29696_c4_2026-10-30_03-42-10.jsonl)

计时修正前的8个预备请求及其命令位于 [旧计时预备记录目录](/Users/wuyidong/projects/newly_job/spec-workspace/specforge/my_docs/2026-10-30_vllm-ascend_MRV2_ctxwin_910b3验证证据/mrv2_long_legacy_timing)，没有纳入32条正式长上下文结果。

## 验证完成时的服务快照

以下仅描述本次验证完成时的状态，不作为当前进程状态：当时保留窗口增训草稿、MRV2、图模式、2048／128窗口服务，/health与/v1/models均返回200。后续接手操作须重新核对PID、服务归属及卡占用。

```text
FINAL_COMMIT=4a01608da7be1950eebc9902e3692d257f304759
FINAL_SERVICE=A+2048/128, MRV2 graph, TP4, port 8861
FINAL_HEALTH={"/health": 200, "/v1/models": 200}
FINAL_API_HOST_PID=283779

CONTAINER PROCESSES
PID                 PPID                ELAPSED             COMMAND
283474              2469327             07:06               bash -lc cd /opt/w00958190; export DRAFT_MODEL=/opt/w00958190/specforge_tmp/export_A_agent_ctxwin2048_step50; unset DRAFT_WINDOW; bash /opt/w00958190/specforge_tmp/run_server_qwen3.8_dspark_mrv2_ctxwin.sh --mode compile --spec-method dspark --spec-steps 7 --max-seqs 8 --tp 4 --gpus last4 --gpu-mem-util 0.9 --max-model-len 32768 --port 8861 > /opt/w00958190/specforge_tmp/logs/serve_V2long.log 2>&1; rc=$?; echo SERVER_EXIT=$rc >> /opt/w00958190/specforge_tmp/logs/serve_V2long.log; exit "$rc"
283764              283474              07:05               bash /opt/w00958190/specforge_tmp/run_server_qwen3.8_dspark_mrv2_ctxwin.sh --mode compile --spec-method dspark --spec-steps 7 --max-seqs 8 --tp 4 --gpus last4 --gpu-mem-util 0.9 --max-model-len 32768 --port 8861
283779              283764              07:05               python -m vllm.entrypoints.openai.api_server --model /opt/models/Qwen3.8-27B --served-model-name qwen3.8-27b-bare --max-model-len 32768 --max-num-seqs 8 --data-parallel-size 1 --tensor-parallel-size 4 --port 8861 --gpu-memory-utilization 0.9 --trust-remote-code --tokenizer-mode auto --tool-call-parser qwen3_coder --enable-auto-tool-choice --reasoning-parser qwen3 --allowed-local-media-path / --enable-request-id-headers --speculative-config {"num_speculative_tokens": 7, "method":"dspark", "model": "/opt/w00958190/specforge_tmp/export_A_agent_ctxwin2048_step50"} --default-chat-template-kwargs {"enable_thinking": false} --enable-prefix-caching --compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[8, 16, 24, 32, 40, 48, 56, 64]} --async_scheduling
285663              283779              06:25               /usr/local/python3.12.13/bin/python -c from multiprocessing.resource_tracker import main;main(53)
285664              283779              06:25               VLLM::EngineCore
286734              285664              06:03               VLLM::Worker_TP0
286735              285664              06:03               VLLM::Worker_TP1
286736              285664              06:03               VLLM::Worker_TP2
286737              285664              06:03               VLLM::Worker_TP3
2469497             2469327             2-01:37:08          /bin/bash

 NPU ID                         : 4
 Chip Count                     : 1

 Process id:283779  Process name:python            Process memory(MB):106
 Process id:286734  Process name:VLLMWorker_TP     Process memory(MB):55536
 Chip ID                        : 0


 NPU ID                         : 5
 Chip Count                     : 1

 Process id:286735  Process name:VLLMWorker_TP     Process memory(MB):55536
 Chip ID                        : 0


 NPU ID                         : 6
 Chip Count                     : 1

 Process id:286736  Process name:VLLMWorker_TP     Process memory(MB):55536
 Chip ID                        : 0


 NPU ID                         : 7
 Chip Count                     : 1

 Process id:286737  Process name:VLLMWorker_TP     Process memory(MB):55536
 Chip ID                        : 0
```
