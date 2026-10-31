2026-10-31

# DSpark 补充实验：方法、命令与技术证据

> 范围更新：用户已确认收敛本轮，原完整矩阵不再全部执行。四组首轮和已启动的 size16 第二轮保留；重点补只读基线、回收 size8、相同分组全量保留三组诊断，以及 W2048 边界/复用和32并发池占用。其余重复轮次、次要训练/窗口组合、eager、W128边界和抢占验证列为后续。

正文结论以补充报告为准；本附件保留复核口径和原始证据位置，不把待完成项目写为通过。

## 证据目录与正式结果边界

本轮正式目录为容器内 `/workspace/dspark_ctxwin_validation/round_20261031_isolated`，以下命令均从该目录运行。服务日志在 `/opt/w00958190/log/`。`round_20261031` 是早期预实验：其 Triton 缓存未按配置隔离，因此保留全部文件但不纳入本轮正式数值。旧版本约 7.17 GiB/卡和 389 项 CPU 回归也不算本轮新结果。

新目录内首个基线的六个普通 `_perf_r1_` 波次另列为冷启动观察；它们暴露了强制输出路径首次编译的时间。正式速度只统计 `_steady_perf_` 文件，明确排除 `_warm_perf_`。首个基线已经重跑六个稳态条件；其缓存文件最后更新时间早于这六波，证据为 `baseline_steady_cache_retrospective.json`。后续条件还记录 `_cache_audit.json` 的前后缓存文件变化。不能用整波结束后的文件列表冒充逐请求编译时间线。

`series_manifest.json`、`STATUS.md` 和 `performance_warmup_correction.json` 保存上述调整。所有旧结果原样保留。生产分支源码本轮未修改。

## 实际运行入口与复现说明

连接 `ssh 910b3`，再进入 `docker exec -it wyd_va_030rc2_1028 bash`。只使用物理设备 4、5、6、7。服务切换必须先核对进程命令、模型名、实际入口和设备环境；下列 driver 已做这些检查，不能改成按名称批量终止。

```bash
cd /workspace/dspark_ctxwin_validation/round_20261031_isolated
python3 -u matrix_driver.py 2>&1 | tee /opt/w00958190/log/dspark_20261031_isolated_matrix_driver_startup2.log
```

以下命令记录本轮实际使用的入口，不是要求把原计划重跑一遍。复跑须使用新的结果tag和缓存目录；不能在当前服务仍运行时另起一份，也不能覆盖历史结果。driver 在同一个前台会话启动、预热、评测和切换服务。原计划重点配置三轮交错、次要配置一轮；实际按用户收敛决定只完成四组首轮及size16第二轮，见scope_reduction.json。此旧driver包含未执行计划，不能据脚本存在认定已完成。服务完成后保持前台等待，不自动释放卡。后续监督入口等待正式矩阵完成，然后依次运行诊断、复用和内存检查：

```bash
python3 -u followup_supervisor.py 2>&1 | tee /opt/w00958190/log/dspark_20261031_isolated_followup_supervisor.log
```

用户收敛范围后，实际续跑入口如下；原计划未执行项以`scope_reduction.json`和阶段日志为准，不能把旧driver中的循环当成完成证据。

```bash
python3 -u resume_reduced_diagnostics.py 2>&1 | tee /opt/w00958190/log/dspark_20261031_resume_reduced_diagnostics.log
python3 -u resume_boundary_poolonly.py 2>&1 | tee /opt/w00958190/log/dspark_20261031_resume_boundary_poolonly.log
```

`serve.sh` 保存完整参数。常规服务固定：MRV2、DSpark、TP4、7 个推测 token、BF16、seed 42、异步调度、FA4 sink=2、关闭 prefix caching、开启 hybrid KV manager、最大长度 32768、最大请求数 8、每批最多 8192 token、显存利用率 0.9。图配置请求 FULL，实际 target 使用 FULL_DECODE_ONLY，draft 使用 FULL。诊断的块池预算和 32 并发配置单独记录，不能与常规速度混算。

每个服务配置分别隔离 VLLM、Triton、Inductor、torch extensions、Ascend、torchair 缓存和工作目录。`startup_audits.json` 汇总每个健康启动实际导入路径、真实分组、块数、缓存路径和已加载探针。普通测速入口只保留初始化审计，不装入逐步采样。

## 源码与二进制

- 只读窗口基线：`4a01608da7be1950eebc9902e3692d257f304759`，目录 `/workspace/dspark_ctxwin_baseline`。仅补充启动分组打印，差异保存在 `baseline_source.diff`。
- 回收分支：`ed48548e885aa37126e497c102124184385c636a`，目录 `/workspace/dspark_ctxwin_tmp`。
- vLLM：`ced6857afa0ea7b2e3f0846a62e1394e90f15607`，目录 `/vllm-workspace/vllm`。
- FA4：`64a5652de4ba158083c43a6d4ed0a474a7a11419`，目录 `/opt/w00958190/flash-attention-npu`。

两个源码入口使用同一份预编译 Ascend 扩展。其路径、提交、大小和 SHA256 保存在 `binary_dependencies_initial.json`；结束前已再次核对，结果见`binary_dependencies_final.json`，源码复核见`source_audit_final.json`。这是复现依赖，不能仅以 pip 显示版本作为源码版本。

两个草稿导出都保持原样。变体只复制 `config.json`，权重软链到原始导出。关闭窗口同时移除两个窗口字段。`runtime_manifest.json`、`source_audit.json` 和 `models/` 中的配置保存输入来源；`models_final_audit.json`额外记录正式隔离目录的实际路径，并核对配置散列及原始权重的设备号、inode、大小、修改时间。没有对多GB权重重新做全文件内容散列，不将文件身份检查称为内容完整性证明。

## 计数、时间和业务口径

原业务评测每轮工具调用 50 题、最终回答 50 题，分别取服务 metrics 前后差值。指标为 accepted/drafted 和 1+accepted/steps；原始计数、每个接受位置计数、有效题数、输出长度和失败数均保留。并发 4、最大输出 1024、temperature=0、top_p=1、关闭 thinking。

性能每个条件先以相同 4 条输入、相同并发、强制 64 token 预热，再强制 512 token。实际渲染后的平均输入分别约 13543.5、24584.75、29708.75 token，以逐题 usage 为准。并发分别为 1 和 4。条件顺序轮换；实际四组各一轮，size16两轮，预热不计统计，未完成三轮重复。

TTFT 从发送请求至收到第一个含 token IDs 的流事件。TPOT=(流结束时刻−首个token到达时刻)/(实际输出token数−1)；一个流包包含多个 token 时仍按实际 token 数计算。整波吞吐包含 prefill 和排队。另存首个非空内容、推理或工具字段到达的辅助时间，它不等于纯正文首包。队列等待、抢占和结束时 running/waiting 从 metrics 差值或期末值核对。

工具调用检查 JSON、外层 schema、函数及 Skill 路由、参数完全匹配；没有执行真实下游工具，所以 schema 通过不能被解释为业务全链路成功。最终回答逐字匹配不等于语义正确。参考本身有疑点的样例需要人工注明。

## 分叉诊断如何避免比较不同前缀

普通入口先保留每条请求完整 token IDs，固定输入、提交顺序、seed，分别在并发 1 和 4 重复。并发 1 仍可能因 async 请求交接形成不同的引擎批次；客户端提交顺序也不能保证并发请求在服务内有完全相同的调度时序。

`build_divergence_prefixes.py` 从本轮正式重复结果选取 EOS 之前的首分叉，冻结实际共同前缀。它还保留基线首个输出 token 即分叉的样例，检查 draft 候选尚未参与输出之前的 target 行为。来源、位置及 token IDs 写入 `official_prefix_selection.json`。

`replay_prefixes.py` 重新 prefill 同一前缀并返回 target top logprob。`analyze_prefix_trace.py` 同时核对实际送入 target 的 token 和位置、分块 prefill 历史、首个 target 候选及图形状。重新 prefill 的 GDN 状态与原增量路径不完全相同，不能单独据此解释原分叉。

原长文轨迹由 `analyze_trace_divergence.py` 对齐：只取实际共同 token 前缀下的 target 验证行、draft 提案、接受位置和有效读块表，排除更早拒绝后未被采用的假设 token 行。诊断代码只在隔离入口，增加的设备复制、top-k 和日志可能改变时序，因此不进入速度表；分叉未复现也不是正确性证明。

同组全量保留对照维持 SlidingWindowSpec、分组和 tensor 形状，验证专用地将跳过数改为 0、取消回收准入上限。实际分组和池块数必须与回收版逐项相同。该方法不改变生产分组算法，不是一个新的生产配置。

## 边界、块复用与内存

W2048/align128/B7/S7 的保留量应为 2191，W128 对应 271。运行配置和探针记录必须印证它。有效读列的物理块 ID、当前 context 写槽和 draft query 写槽分别检查；896-token 调度块展开成 128-token kernel 块后，null 判断必须还原物理块 ID。

回池记录包含物理块 ID、组、请求、引用计数和空闲池计数。复用证据需要串起：原请求窗口归还 → 其他请求领取同一个块 → 原请求继续产生输出。请求退出和取消的延迟释放须结合 async 在途计数解释。

池占用分三项：实际持有的共享槽字节、有效层页字节、两者差额即组内空位。页内对齐 padding 已计入 page，不能声称这是有效 token 的理论最低值。npu-smi 进程显存和启动最大并发估算均不能代替运行占用。

32 并发分别测只读基线、同组全量保留、回收 size8；最后两者强制相同块池大小。表中逐一列实际分组。来自 null 前缀列的反事实回收量作为补充计算，同时提供真实全量保留对照，不能把反事实估算写成直接观测。

## 证据文件索引

- `matrix_plan.json`、`matrix_progress.jsonl`：完整配置、顺序、阶段和异常。
- `results/*jsonl`、对应 `.metrics.0.txt`/`.metrics.1.txt`、`.summary.json`：逐题输入输出 token、计时和服务计数。
- `aggregate_results.json`、`business_token_comparisons.json`：重复统计和逐题分叉。
- `report/business_rounds.csv`、`report/performance_rounds.csv`：可读原始轮次表。
- `results/*fixed*.json`、`results/*prefix_replay*.json`：固定 token 及同前缀重放。
- `results/*pool_summary.json`、映射与归属分析文件：真实物理块、边界和复用。
- `/opt/w00958190/log/dspark_20261031_diag_*.jsonl`：各进程完整诊断轨迹。
- `retained_service_health.json`：本任务服务最后一次成功的实际请求与HTTP健康检查；`retained_service_audit.json`记录其后关闭和另一验证任务接管的只读观察，不能把前者当成当前服务状态。

便携证据位于仓库`mydocs/dspark_followup_20261031_evidence/`：`remote_results_sha256.json`索引原始结果，`validation_harness.tar.gz`归档验证脚本、隔离配置和首次异常日志，`formal_per_request_results.jsonl.gz`保留620条正式业务/速度请求结果及输出token IDs。完整请求模板、输入IDs和大体积设备轨迹保留远端；便携副本以输入散列与原始文件路径对应，避免重复存储。

本地机器与容器时钟约相差298秒（容器较慢），只做了只读核对，未调整系统时间。性能耗时使用容器内客户端单调计时；scheduler与worker轨迹使用相同容器时钟。跨机器日志对齐应参考clock_observation.json，不能直接把本地显示时间与服务器时间相减。

## 诊断入口的两项修正（生产代码未改）

第一组设备轨迹的CPU时间归属检查曾失败：49,560次物理读块检查和18,256次物理写块检查中，372次读、108次写的采样时刻已查不到CPU池归属。逐项回看，480条均对应同一请求已FINISHED、相同请求/组通过free归还、且采样时没有新主人领取该块；属于请求末尾仍排入设备流的异步步骤。原失败报告保存在`*_ownership_summary.original_failure.json`。

上游`vllm/v1/core/sched/scheduler.py`的`_request_blocks_can_be_freed`说明，普通设备流内操作与可能立即写入的异步connector必须分开处理。本轮无connector。修正后的检查保留全部CPU时间不一致观测，仅把上述有明确结束证据且尚未重新领用的情况单列为terminal_release；其他不一致仍失败。合成回归分别验证：末尾释放通过，窗口回收、重新领用、缺少FINISHED证据三种情况仍失败。不能把这项分类写成“设备硬件完成时刻已经逐步证明”。后续真实复用另行验证。

分叉分析还修正了请求ID匹配：原来的子串匹配可能把`archive_29696`与`archive_29696_at_88`的同前缀重放混在一起。现在只匹配完整completion ID及运行时实际附加的索引/十六进制后缀。合成回归排除重放ID及长前缀碰撞。四条原始长文每侧均找到唯一实际采样行，所选token与客户端首分叉一致；原始轨迹不需重跑。

第一组重新prefill的12次前缀重放均覆盖所有输入位置、无输入token不一致。原始增量长文的四次自对照首分叉分别在输出索引26、440、88、51；target最高候选的分数差异及并列情况已记录。分叉处均为单请求FULL形状8；前三条当步target输入、前一步GDN接受长度也一致，第四条前一步接受长度为4与5。它们定位到target验证决策发生变化，尚未定位产生数值差异的具体算子。三组诊断现已完成；results/diagnostic_overview.json及results/incremental_divergence_case_summary.json汇总37个首分叉配对记录。它们含自对照和交叉对照，不是37道独立题目。

## 长短请求混跑中的设备异常（尚未归因）

`boundary_isocache_trained_window_W2048_reclaim8_pool384`在10条边界请求完成后运行取消、错峰进入、两条长输出与多波短请求混跑。第10波短请求附近停止前进；服务器UTC 07:26:17先出现设备页错误，07:35:32报告`FusedInferAttentionScore`内核AICore timeout/trap，进程退出。设备日志、栈和池分析保存在`failure_20261031_boundary_trace/`与`boundary_stall_*`。最低空闲53块、无抢占、已记录池守恒及保留边界检查无违规，不能因此排除设备读写问题。进程栈停在`vllm_ascend/worker/v2/model_runner.py:1348`的`_update_seq_lens_cpu`等待设备同步；这是等待位置，不能当作出错算子的源码位置。

原trace模式失败不能算作复用验收通过。为区分额外设备采样的影响，另用`boundary_retry_poolonly_trained_window_W2048_reclaim8_pool384`复跑相同边界与混跑负载，仅保留CPU BlockPool记录；尚未据此认定采样就是原因。完整启动记录在`/opt/w00958190/log/dspark_20261031_resume_boundary_poolonly.log`。首次失败混跑的客户端在整波结束前未逐题落盘，最后待flush的设备轨迹也未写出；现有证据不能补出故障瞬间完整映射。恢复时修改验证客户端：每条完成或失败请求立即追加journal，防止最终文件未写出时丢失逐题结果；旧脚本保留为`boundary_bench.before_incremental_journal.py`。生产源码未改。

旧服务已自行退出后，恢复入口只在确认没有本任务残留服务/worker时允许启动；仍不操作其他任务进程。恢复脚本`resume_boundary_poolonly.py`继续本轮收敛计划，不重复已完成的正式业务与速度矩阵。

## 复测后的边界与复用结果

仅保留CPU池记录的重跑完成10条边界请求及40条混跑请求（1条预期取消，39条正常完成）。窗口归还169块次，65次跨请求窗口块复用后原请求继续输出，其中21次发生于原请求已开始输出之后，涉及19个物理块；其中12次来自两条各6000token的持续长请求。最终空闲恢复383/383，最低空闲53，无抢占或池守恒违规。`results/boundary_poolonly_outcome.json`附可追踪的旧请求、新请求、物理块ID、归还/重新领取时间以及旧请求后续完成记录。

首次带设备采样的边界阶段独立完成：有效读列108304、context写槽309364、query写槽44744，128对齐边界变化176次、896调度块边界变化36次、长度截断步骤16次；这些包含四个TP rank重复观察。其后混跑失败，因此这组映射检查不能覆盖失败时刻。关闭设备采样的复测也不能替代每一步设备映射检查，原AICore异常根因仍待定位。

`max_inflight=16372`包含分块prefill的乐观在途计数，不能直接代入“scheduler回收点与draft anchor相差多少”的安全公式；回收点使用的computed已按真实调用路径扣除对应在途量，原推导见保留策略报告。

## 32并发内存的实际完成结果与设备状态

三组各两波、每波32条31744输入+1024输出请求全部完成，共192条。真实分组/池大小从scheduler配置读取；samegroups_full8与reclaim8均为5616块，逐项一致。窗口归还仅统计请求ID带本轮`_wave`的事件：只读基线0、同组全量保留0、回收2176块次。结束后空闲块分别恢复2138、5615、5615，均等于总池减一个null块。

`compare_memory32.py`仅使用32条请求同时运行、全部扣除在途量后完成prefill且已有输出的快照。按target FullAttention列数和GDN层页数分层，对齐33个共同分配状态；不使用npu-smi显存。最大共同状态为1184列和12288个GDN层页。三组实际持有池槽分别为35.02734375GiB、34.78125GiB、27.5146484375–27.5625GiB；回收观察中位27.5556640625GiB。净节省中位7.4716796875GiB，有效层页减少4.5159912109375GiB、组内空位减少2.9556884765625GiB。相同size8布局下回收贡献中位7.2255859375GiB，布局贡献0.24609375GiB。

三个配置的每波稳定区间、原始字节值、全部共同状态和样本量见`results/memory32_comparison.json`。每个快照的分组占用、窗口归还及复用链仍保存在各自`*_pool_evidence.json`。共同状态的同组全量保留实测值与回收轨迹的历史null列反事实逐项相等，这项一致性由脚本断言检查；反事实没有替代真实全量保留运行。

`memory32_token_comparisons.json`保存9组各32条逐token配对及首分叉、共同前缀、首次EOS位置；包含三组自重复以及同波跨配置比较。全部配对的完整一致数为0。固定输入、输出长度不等于固定target数值轨迹。

故障后的只读检查`npu-smi info -t health -i 4/5/6/7 -c 0`均报告Alarm、80C98001，描述为AIC / RAS State / module error can not be fixed。原文保存在`device_health_after_exception.json`。这不是“告警已清除”的状态；没有据此推断硬件永久损坏，也没有擅自做宿主机或设备级复位。正式无探针测速发生在该异常之前；复用重跑及三组内存测量发生在异常之后，二者不混为同一设备状态的性能测量。

## 交付文件与服务交接

正文的服务交接节记录本任务健康请求成功后发生的关闭与另一个验证任务启动。未干预接管任务；其时点状态见`retained_service_audit.json`，本任务最后成功请求仍保留在`retained_service_health.json`。设备Alarm单独记录，未写成恢复。

便携附件包括业务/速度CSV、620条正式逐请求压缩结果、分叉与内存摘要、源码/二进制/模型配置核对、原始结果SHA256索引、验证脚本归档。服务器完整证据根目录及历史日志继续保留。文档读者检查已覆盖业务、分叉、速度、内存分解和异常边界；修正了请求数、比较方向、准入策略及TPOT计时口径等歧义。格式检查记录在`repository_format_check.json`；完整格式入口缺pre-commit，本轮未升级共享环境。
