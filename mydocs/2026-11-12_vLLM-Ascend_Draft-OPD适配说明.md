# 2026-11-12 vLLM-Ascend Draft-OPD适配说明

本文对应 `fg11991/vllm-ascend` 的 `030_dspark_opd_trace` 分支，代码固定到 **`ce9ca72d19c43bf6c6fce02f6f5dc2fc3958371d`**。配套训练代码为 SpecForge `codex/qwen38-dspark-first-reject-opd` 的 **`a99ab23db13f3831e63959d71211b4b2bf9d7249`**。下述源码行号均以这些提交为准，后续分支更新可能使行号变化。

我们的 vLLM-Ascend 适配承担的是**真实推理轨迹的可靠采集**：记录调度器实际提交了哪些 token、这轮从哪里开始、实际验证多少位、接受多少位，以及使用的是哪份草稿权重。SpecForge 据此重建首拒位置并训练。KL 损失、teacher hidden 抓取和优化器更新均在 SpecForge；此分支没有在推理服务中加入在线梯度更新。

论文原理另见同目录的 [Draft-OPD 论文原理详解](2026-11-12_Draft-OPD论文原理详解.md)。本文重点回答“我们改了什么、为什么必须改、产出的数据能支持什么”。

## 1. 固定版本与移植边界

### 1.1 NAIE 基线与五个增量提交

本地 Git 对象核验得到：`origin/030_naie_dev` 与 `singzhou/030_naie_dev` 在本次核对时均指向 `7f93f07236def39e26b491798c7c461a552c2b23`；它也是 `ce9ca72d1` 与 NAIE 分支的共同基线。基线上已有 DSpark 推理、V2 adaptive verification、相关 GDN 内核和性能改进；这些不能算作本次 OPD 新增功能。

从该基线到目标提交，恰好是**四个 OPD 移植提交，加一个 NAIE 日志入口修复**：

| 顺序 | NAIE 分支上的提交 | 原始 OPD 提交 | 本次增量的作用 |
|---|---|---|---|
| 1 | [`07483adf5763`](https://github.com/fg11991/vllm-ascend/commit/07483adf5763b1b7c59dbcf628e1f3a727344a2a) | `a16314b68586eff03e22665e3b00b3af4c215a33` | 新增显式启用的 sync/async trace scheduler，从实际提交区间记录验证轨迹。 |
| 2 | [`32de0ed0d848`](https://github.com/fg11991/vllm-ascend/commit/32de0ed0d848af1e2de98f12318744734e6e4f5b) | `27fa0a72212207cb761d3d17bc10dcbf56efdba0` | V2 从设备端 sampler 边界读取自适应验证后的真实宽度，避免把预算截断误判为拒绝。 |
| 3 | [`e37ab0bdf603`](https://github.com/fg11991/vllm-ascend/commit/e37ab0bdf60397659736fa8030cc127b32ca0c0e) | `5fe0363bf8f9e38b5763847dc19571f9bcafb21e` | 检查最终生效的 scheduler，绑定实际草稿文件指纹，补足停止语义；在采样复用缓冲区前读取宽度。 |
| 4 | [`638708eba6fc`](https://github.com/fg11991/vllm-ascend/commit/638708eba6fcbd5483f993d33b90774fced726e6) | `9434a131db6ac6ebb0824fcab2146382d16bad11` | 按 V1/V2 实际采用的加载器配置校验权重来源，避免验证了未生效的 draft override。 |
| 5 | [`ce9ca72d19c4`](https://github.com/fg11991/vllm-ascend/commit/ce9ca72d19c43bf6c6fce02f6f5dc2fc3958371d) | NAIE 上新增修复 | 改用已配置的 `vllm.logger.logger` 输出 trace，并同步修改测试与 import 顺序。 |

四个移植提交的 Git 提交说明中均保留 `cherry picked from commit ...`，因此来源可追溯。整个增量只有 **6 个文件、345 行新增、1 行删除**，其中 152 行是独立单元测试。

### 1.2 六个修改文件

| 路径 | 职责 |
|---|---|
| [`vllm_ascend/core/opd_trace.py`](../vllm_ascend/core/opd_trace.py) | trace scheduler、请求逐轮事件、真实验证宽度采样包装。 |
| [`vllm_ascend/core/opd_trace_config.py`](../vllm_ascend/core/opd_trace_config.py) | 识别 OPD scheduler，以及检查平台设置后是否仍然有效。 |
| [`vllm_ascend/core/opd_draft_digest.py`](../vllm_ascend/core/opd_draft_digest.py) | 按张量内容计算本地 HF safetensors 草稿的 SHA-256。 |
| [`vllm_ascend/platform.py`](../vllm_ascend/platform.py) | 平台改写配置前保存 scheduler，改写后检查是否被覆盖。 |
| [`vllm_ascend/worker/v2/model_runner.py`](../vllm_ascend/worker/v2/model_runner.py) | 在 V2 sampler 真正采样前取得设备端有效验证宽度。 |
| [`tests/ut/core/test_opd_trace.py`](../tests/ut/core/test_opd_trace.py) | 不初始化 NPU 的 scheduler double 测试。 |

固定版本源码入口：[trace 实现](https://github.com/fg11991/vllm-ascend/blob/ce9ca72d19c43bf6c6fce02f6f5dc2fc3958371d/vllm_ascend/core/opd_trace.py#L16-L114)、[V2 接入](https://github.com/fg11991/vllm-ascend/blob/ce9ca72d19c43bf6c6fce02f6f5dc2fc3958371d/vllm_ascend/worker/v2/model_runner.py#L149-L157)、[平台检查](https://github.com/fg11991/vllm-ascend/blob/ce9ca72d19c43bf6c6fce02f6f5dc2fc3958371d/vllm_ascend/platform.py#L552-L593)。

## 2. 为什么最终文本和公开 metrics 不够

论文的训练信号来自草稿实际访问过的验证位置。如果只保存最终生成文本，那么首个拒绝候选已被 target 的恢复 token 替换，被拒后缀也被丢弃。文本本身无法告诉训练端某轮草稿从哪里开始、哪一位首次失败。

公开的 `per_step_drafted`、`per_step_accepted` 还存在一个工程问题：它们只记验证轮，普通 decode 不一定出现在数组中。两个请求即使有相同的最终 token IDs 和相同的逐步验证数组，普通 decode 穿插的位置仍可能不同，对应 anchor 因而不同。不能靠累计 `accepted + 1` 猜整段输出的位置。

另外，V2 自适应验证可能把“计划提议 7 位”缩成“实际只验证 2 位”。如果两位均接受，把公开 `(drafted=7, accepted=2)` 直接当作第三位拒绝，会制造错误训练标签。真实标签必须知道这轮实际验证宽度。

本次适配因此选择记录**调度器已确认提交的 token 增量**，并补充真实验证宽度。它输出的是训练数据契约，而不是另一套 speculative decoding 算法。

## 3. 调度器如何取得可信事件

核心实现是 `_OPDTrace.update_from_output`，位于 `vllm_ascend/core/opd_trace.py:59`。

一次更新分成三步：

1. 更新前，遍历 `scheduler_output.num_scheduled_tokens`，只对 `sampling_params.extra_args` 中有 `opd_trace_id`、且有 detailed metrics 的请求保留引用，记录当前 `num_output_tokens` 与验证数组长度。
2. 调用原来的 `super().update_from_output(...)`。token 接受、恢复、停止、EOS 和调度记账仍由原调度器完成。
3. 更新后，用 `req.output_token_ids[offset:]` 读取真正提交的增量，写一条 `DSPARK_OPD` JSON 事件，并原样返回原调度器结果。

保存请求对象引用很重要：原调度器可能在这次更新中把完成请求从 `self.requests` 清掉。若更新后再按 ID 查询，请求最后一轮就可能丢失。代码读取的是原调度器已经按停止条件裁剪的 IDs，所以既不把原始 sampler 未提交槽位写成输出，也不把 stop-trimmed token 继续计入输出偏移。

这轮是否发生验证，通过 detailed 数组长度是否恰好增加 1 来判断。普通 decode 的事件仍记录 `offset` 和 `token_ids`，但 `drafted`、`accepted`、`verified_width` 为 `null`。没有新增提交 IDs 的更新不输出事件。

两个显式入口分别继承原同步和异步调度器：

```text
OPDTraceScheduler      = _OPDTrace + Scheduler
OPDTraceAsyncScheduler = _OPDTrace + AsyncScheduler
```

代码检查 scheduler 类型必须与 `async_scheduling` 一致，避免把一种调度模式的记账约定套到另一种模式上。它不是可直接与任意自定义 scheduler 叠加的通用插件；现场若有自定义调度器，须核实继承和最终配置。

## 4. Trace 格式、计数与索引

### 4.1 事件字段

日志一行的有效载荷为 `DSPARK_OPD {JSON}`。前面的 vLLM 日志时间、进程名等前缀由既有日志系统决定。

| 字段 | 实际含义及使用方式 |
|---|---|
| `request_id` | 引擎内部请求 ID；同一条请求的事件须保持一致，不要求它等于 API 返回 ID。 |
| `trace_id` | 采集端通过请求体 `vllm_xargs.opd_trace_id` 提供、进入引擎后存于 `sampling_params.extra_args` 的关联标识，用来把服务日志与客户端响应配对。 |
| `draft_revision` | 启动时的人工版本标签，来源为 `SPECFORGE_OPD_DRAFT_REVISION`。标签用于关联，张量指纹负责验证内容。 |
| `sample_from_anchor` | 来自 draft HF 配置，缺省为 `true`；训练回放必须保持相同 query 槽布局。 |
| `offset` | 本次 scheduler 更新之前已经生成的 **output token 数**，不含 prompt。第一条通常是 0。 |
| `token_ids` | 本次更新后真正新增的 committed output IDs，已按停止条件裁剪。 |
| `drafted` | detailed metrics 中本验证轮的原始 drafted 计数；普通 decode 为 `null`。AV 下不能直接拿它当实际验证宽度。 |
| `verified_width` | 真正进入验证的 proposal 数，不含 target 槽。无 AV 时取 `drafted`；V2 AV 时取 sampler 的设备边界。普通 decode 为 `null`。 |
| `accepted` | detailed metrics 中本验证轮接受的草稿 token 数，不含恢复 token 或全接受后的 bonus token。普通 decode 为 `null`。 |
| `invalid` | 本轮 scheduler 标记的 invalid speculative token 数，缺省为 0；训练端排除 grammar 调整轮，避免把约束裁剪当成模型错误。 |
| `preemptions` | 请求累计抢占次数；当前 SpecForge pack 拒收有抢占的请求。 |
| `stopped` | 原调度器更新后请求是否结束。完整训练块排除终止轮，不能把提前停止当成一次拒绝。 |
| `stopped_on_eos` | 更窄的 EOS 判断：请求结束、状态为 `FINISHED_STOPPED`、没有额外 `stop_reason`，且最后提交 ID 等于 `eos_token_id`。它不等价于所有可能的停止原因。 |
| `prompt_digest` | 仅 `offset==0` 事件携带；对 prompt ID 列表用紧凑 JSON 序列化后求 SHA-256。用于校验服务实际 prompt IDs 与响应中的 prompt IDs。 |
| `draft_weights_sha256` | 仅 `offset==0` 事件携带；对本地实际草稿文件的规范张量内容求指纹。 |

`token_ids` 是最终提交的真实 IDs，**不是完整原始 draft proposal**。这个区别决定了当前数据只能直接支持接受位置与首拒位置的完整分布蒸馏，无法凭这些事件重建首拒之后的原始草稿前缀。

### 4.2 output offset 到绝对 anchor

设实际 prompt 长度为 `P`，某个完整验证轮事件的 `offset=c`，采用整段 `prompt + output` 的 0-based 索引，则：

$$
a=P+c-1.
$$

`a` 是第一位候选之前的 token 位置。设实际验证宽度 `d=verified_width`，接受数为 `r`：

- 接受候选对应序列位置 `a+1` 到 `a+r`。
- 若 `r<d`，首拒对应位置 `a+r+1`；监督这个位置使用 teacher 在 `a+r` 的最后状态。
- 若 `r=d`，这轮没有首拒监督。
- 完整非终止轮通常提交 `r+1` 个 token：`r` 个接受 token，加 target 恢复 token，或全接受后的 bonus token。

示例：`P=10`、`offset=5`、实际验证 7、接受 2，则 anchor 为 14，位置 15 和 16 已接受，位置 17 是首拒位置。最终 output 在位置 17 写的是 target 恢复 token，不是原被拒候选；首拒 KL 比较发生在双方仍共有的前缀 `...16` 上。

再如计划 7 位、V2 AV 实际验证 2 位、接受 2 位：标签必须是 `[a,2,2]`，不应是 `[a,7,2]`。两位都通过，没有被实际验证过的第三位，也没有第三位首拒损失。

### 4.3 普通 decode 必须推进 offset

单元测试使用四个提交事件说明这个约定：

| 事件 | `offset` | 提交 IDs | 原始 `(drafted,accepted)` | `verified_width` |
|---|---:|---|---|---:|
| 首次普通 decode | 0 | `[3]` | `(null,null)` | `null` |
| 验证轮 | 1 | `[1,2]` | `(7,1)` | 2 |
| 中间普通 decode | 3 | `[6]` | `(null,null)` | `null` |
| EOS 终止验证轮 | 4 | `[8]` | `(7,3)` | 7 |

最后一轮 metrics 的 accepted 数与实际提交长度可因终止裁剪不再满足 `len(ids)=r+1`；因此 pack 仍保留这条输出作为上下文和 metrics 校验依据，但将终止轮排除出完整首拒训练块。这个例子来自 scheduler double，作用是测试记账契约，不是实际模型生成样本。

## 5. V2 自适应验证宽度的适配

### 5.1 读取最终设备边界

`sample_with_verified_widths` 位于 `opd_trace.py:16`。它先读取：

```python
offsets = batch.cu_num_logits[: len(batch.req_ids) + 1].tolist()
widths = {rid: end - start - 1
          for rid, start, end in zip(batch.req_ids, offsets, offsets[1:])}
```

相邻边界之差给出这条请求交给 sampler 的 logits 行数；减掉 1 个 target 槽后，才是实际 proposal 验证宽度。

这一步使用设备端 `cu_num_logits`。AV 后 CPU 镜像可能仍是预分配上界，代码不以 `cu_num_logits_np` 推断真实宽度。测试故意让两者不同，证明输出依据的是最终设备边界。

### 5.2 必须在 sampler 复用 batch 前做快照

随后函数调用原来的 `sample_tokens(grammar_output)`，把宽度字典附到 `ModelRunnerOutput.opd_verify_widths`；异步输出若套有 `model_runner_output` 包装，也会写入内部结果对象。

宽度在调用 sampler **之前**读取，因为 batch 的存储可能在采样期间复用。测试中的模拟 sampler 会清零原来的 `cu_num_logits`；最终仍应得到 `{x:0,y:7,internal-17:2}`。

`NPUModelRunner.__init__` 只有在“存在 speculative config、启用 AV、有效 scheduler 是 OPD scheduler”时打开 `_trace_opd_widths`。`sample_tokens` 还要求有执行状态、处于最后一个 pipeline rank，才走该包装，其他情况调用原采样入口。

这里增加一次设备到主机的 `.tolist()` 读取，可能产生同步开销。没有 NPU 性能实测时不能给出其吞吐影响数值。即便某个请求不带 trace 标记，只要服务选中了 OPD scheduler 且启用该 AV 路径，宽度包装仍是服务级启用；不能把“未标记请求不输出 JSON”理解为“该请求完全没有额外工作”。生产吞吐应恢复原 scheduler、关闭 trace 后单独测量。

固定源码：[宽度包装](https://github.com/fg11991/vllm-ascend/blob/ce9ca72d19c43bf6c6fce02f6f5dc2fc3958371d/vllm_ascend/core/opd_trace.py#L16-L23)、[采样接入](https://github.com/fg11991/vllm-ascend/blob/ce9ca72d19c43bf6c6fce02f6f5dc2fc3958371d/vllm_ascend/worker/v2/model_runner.py#L377-L401)。

## 6. 防止“看起来开了，实际没生效”

### 6.1 最终 scheduler 检查

Ascend 平台初始化可能根据其他配置重写 scheduler。`platform.py` 在平台处理前保存 `requested_scheduler`，在 `_setup_worker_and_scheduler` 后调用 `validate_opd_scheduler(requested,effective)`。

识别函数同时接受字符串形式和实际类对象。若用户显式选择 OPD scheduler，而最终变成普通 scheduler 或其他 Ascend scheduler，启动就报出 `Requested OPD trace scheduler ... was replaced by ...`，避免服务成功启动却一直没有轨迹。

这项检查只保证有效配置仍属于两个支持的 OPD scheduler 名称；它不代替所有现场自定义调度逻辑的兼容性验证。

### 6.2 草稿权重内容指纹

人工 revision 可以写错，也可能同名目录被覆盖。`safetensors_digest` 在 scheduler 初始化时读取本地 HF 草稿，要求目录中存在且只存在一种规范布局：单个 `model.safetensors`，或 `model.safetensors.index.json` 加其准确列出的 shards。

指纹计算分两层：

1. 每个张量保存 `[dtype,shape,SHA256(raw_bytes)]`。张量先转换为 CPU contiguous，再按原始字节求摘要。
2. 按张量名称排序，把上述字典用固定 JSON 形式序列化后，再求一次 SHA-256。

因此同一组张量重新分片不会改变指纹；改动名称、dtype、shape 或值则会改变指纹。索引与文件不一致、目录中夹带额外 safetensors、索引缺失张量或内容为空均会报错。

这个指纹证明的是**草稿文件内容**，不覆盖 target 权重、共享自 target 的 embedding/head、tokenizer、聊天模板、运行时量化和其他部署处理。完整实验仍需单独固定这些配置。离线量化若改变张量内容，指纹也会改变，不能只靠同一个人工 revision 强行当成同一份可训练起点。

固定源码：[指纹实现](https://github.com/fg11991/vllm-ascend/blob/ce9ca72d19c43bf6c6fce02f6f5dc2fc3958371d/vllm_ascend/core/opd_draft_digest.py#L9-L43)。

### 6.3 V1/V2 实际 loader 的区别

`638708eb` 修正的是“该检查哪一份 load config”：

- V2 DSpark 采用全局 `vllm_config.load_config`。
- V1 若有 `spec.draft_load_config`，采用 per-draft override；否则也用全局配置。

代码允许 `auto`、`hf`、`safetensors`、`fastsafetensors`、`instanttensor`，拒绝 `dummy` 等无法支持文件内容身份的加载方式。它不能简单地写成 `draft_load_config or global_config`，否则 V2 上可能检查一份实际没有生效的 override。

允许这些 loader 名称不等于能从任意远程模型标识取得指纹；当前指纹函数仍要求可访问的本地规范 safetensors 目录。

## 7. NAIE 日志入口改动

`ce9ca72d1` 的核心修改只有：

```python
# 移植后的原实现
from vllm.logger import init_logger
logger = init_logger(__name__)

# NAIE 基线上本次采用的实现
from vllm.logger import logger
```

trace 现在直接走已配置的 vLLM logger，其输出位置、格式、handler 等沿用服务日志配置。对使用定制日志配置的 NAIE 服务，这能使 `DSPARK_OPD` 事件进入既有 vLLM 日志入口，便于前台 `tee` 捕获。代码本身没有增加独立文件 logger、后台日志进程或另一套日志配置。

初始化还检查 `logger.isEnabledFor(logging.INFO)`，因为事件采用 `logger.info(...)`。若 INFO 未启用，直接报错；否则服务可能有正常响应但没有采集日志。实际 handler/filter 仍由现场配置负责，启用后还应抽查文件中确实出现 `DSPARK_OPD`，不能仅凭启动成功判断采集完整。

测试相应把 `vllm.logger` 替身由 `init_logger=logging.getLogger` 改为 `logger=logging.getLogger("vllm")`。另一项改动只是将 `is_opd_scheduler` 的 import 放到合适位置。这个提交没有修改事件字段、accepted 语义或训练损失。

固定 diff：[NAIE 日志修复](https://github.com/fg11991/vllm-ascend/commit/ce9ca72d19c43bf6c6fce02f6f5dc2fc3958371d)。上述结论由代码差异可直接确认；本次文档核对未取得 NAIE 远端日志，不能据此声称已完成实机日志链路验收。

## 8. 如何与 SpecForge 对接

数据流为：

```text
真实请求 + 当前 draft
        ↓
vLLM-Ascend 原有 draft / verify / commit
        ├── API 响应：真实 prompt/output IDs、公开 detailed metrics
        └── DSPARK_OPD：实际 offset/提交IDs/真实宽度/accepted/权重指纹
                         ↓
SpecForge collect + pack：逐轮对拍 → [absolute_anchor, width, accepted]
                         ↓
prepare_hidden_states：按原始 token IDs 抓 target hidden
                         ↓
DSpark warm start：原训练目标 + 接受位 forward KL + 首拒位 reverse KL
                         ↓
导出新 draft，下一轮重新采集并在留出会话上比较
```

SpecForge `verify_response` 不直接信任日志。它要求 offset 从 0 连续推进、每轮 token IDs 与 API 输出完全一致、prompt 指纹相同、revision 和 sample layout 一致，验证轮原始计数与公开 detailed 数组逐项相同，并检查 `0≤accepted≤verified_width≤7`。抢占请求被拒收，grammar 调整轮排除，终止轮不成为完整训练块。

对于正常完整轮，得到 `opd_blocks=[P+offset-1,verified_width,accepted]`。此前的普通 decode 和被排除轮提交的 IDs 仍保留在后续上下文里，不能为了“清洗”数据把这些 IDs 从序列中删掉。

配套固定源码：[SpecForge replay 校验](https://github.com/fg11991/specforge/blob/a99ab23db13f3831e63959d71211b4b2bf9d7249/specforge/offline_capture/dspark_replay.py#L76-L155)。

### 8.1 为什么首拒训练不需要记录被拒 ID

在第一次拒绝之前，草稿块已接受前缀与最终 target 轨迹一致。首拒位置的 teacher 与 draft 完整分布均可在这个公共前缀上重算；KL 不需要把恢复 token 冒充被拒候选。因此当前协议可以只存真实提交 IDs 与首拒边界。

若扩展为论文对整个 rejected suffix 的训练，首拒之后草稿前缀已经与最终提交轨迹分叉；这时必须另存或精确重现原 draft proposal，并让 teacher 在这些草稿前缀上重新打分。本分支没有采集这些被拒后缀，不能声称已复现论文所有 rejected token 的 replay。

## 9. 启用方式与现场核对

以下为人工执行的接入示例，不代表本文已经启动或切换服务。沿用既有模型路径、TP、量化、模板和所有服务参数；只替换对应 scheduler、启用 detailed metrics，并记录日志。现场开发和测试在 `910b3` 的 `wyd_va_030rc2_1028` 容器内进行，仅用物理设备 4、5、6、7，遵循已核实的容器设备映射。

```bash
export SPECFORGE_OPD_DRAFT_REVISION="$DRAFT_REV"
export VLLM_LOGGING_LEVEL=INFO

# BASE_SERVE 是原服务完整命令组成的 Bash 数组；已有同名参数应替换。
# 同步服务改用 OPDTraceScheduler，异步服务用下列类。
SCHED=vllm_ascend.core.opd_trace.OPDTraceAsyncScheduler
LOG=/opt/w00958190/log/opd-collection.log
"${BASE_SERVE[@]}" \
  --per-request-spec-decode-metrics detailed \
  --scheduler-cls "$SCHED" 2>&1 | tee "$LOG"
```

OpenAI chat 请求体须携带 `"vllm_xargs":{"opd_trace_id":"unique-id"}`；进入引擎后对应 `sampling_params.extra_args["opd_trace_id"]`。配套工具可代填标记并保存原始请求和响应：

```bash
python -m scripts.dspark_opd collect \
  --url "$URL" --requests train-requests.jsonl --trace --concurrency 1 \
  --output train-responses.jsonl
python -m scripts.dspark_opd pack \
  --responses train-responses.jsonl --log "$LOG" \
  --draft-revision "$DRAFT_REV" --output train-replay.jsonl
```

这两条命令在 SpecForge 目录运行。采集器和训练脚本的完整说明见 SpecForge 的对应适配文档。正式训练前至少抽查真实 prompt IDs、连续 offset、EOS/长度截断、工具文本回包以及存在 AV 时的实际宽度；模型参数、版本标签和文件指纹应写入同一份实验记录。

测试需要切换服务时，应先记下原服务启动命令；结束后恢复后四卡上的可用 vLLM 服务，以 `/health` 和真实生成请求验证。服务持续前台运行，日志写 `/opt/w00958190/log/` 并 `tee` 到终端。本文档核对没有停止、重启或释放任何服务或设备。

## 10. 当前支持边界

| 项目 | 当前实现 |
|---|---|
| 模型方法 | 仅 `method=dspark`。 |
| 额定 proposal 数 | 必须为 7；AV 可实际验证 0–7 位。额定七位与训练 checkpoint 的 `block_size` 不是同一个参数。 |
| 同步/异步 | 各有显式 scheduler，必须匹配服务实际模式。 |
| 按 batch 改 proposal query 宽度 | `num_speculative_tokens_per_batch_size` 非空时报错，未支持。 |
| AV | 要求 V2 runner；从设备读取最终 sampler 宽度。 |
| V1 async + dynamic DSpark verification | 明确拒绝，因为没有可靠的实际宽度遥测。 |
| 请求范围 | 仅带 `opd_trace_id` 的请求输出 JSON；OPD scheduler 的启动校验、指纹和服务级 AV 宽度读取仍会生效。 |
| 权重布局 | 要求本地 HF safetensors 单文件或完整分片索引。 |
| 训练信号 | 现有事件足够支持接受位置与首拒位置完整分布 KL；未记录被拒后缀。 |
| 在线训练 | 无服务内优化器更新或实时换权重闭环；当前按批次采集、离线抓 hidden、续训后重新采集。 |
| NPU 性能影响 | 有潜在宽度同步和 JSON 日志开销，需关 trace 的独立对照实测。 |

## 11. 验证证据与尚未完成的验收

### 11.1 本次实际完成的检查

2026-11-12 在目标 worktree、HEAD `ce9ca72d1` 上执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  --noconftest -p no:cacheprovider -q tests/ut/core/test_opd_trace.py
```

结果为 **5 passed in 0.54s**。执行环境为本地 Python，torch 2.8.0；测试通过模块替身避免初始化 NPU runtime。五项来自 sync/async × trace 开关的四个参数组合，以及一项平台 scheduler override guard。

测试实际覆盖：真实提交区间与连续 offset、普通 decode 与验证轮穿插、请求结束时移出字典后仍能记录最后事件、EOS 标记、prompt/权重摘要、trace opt-in、INFO/revision 缺失、V1/V2 loader 选择、AV 设备边界读取及采样后缓冲区复用、字符串与类对象形式的 scheduler 识别、被平台替换时拒绝。

固定测试：[tests/ut/core/test_opd_trace.py](https://github.com/fg11991/vllm-ascend/blob/ce9ca72d19c43bf6c6fce02f6f5dc2fc3958371d/tests/ut/core/test_opd_trace.py#L16-L152)。

### 11.2 已有文档证据的范围

SpecForge 在 `a99ab23d` 提交内的适配记录列出了 CPU 小模型的三臂 smoke、组合回归和 scheduler double，并明确写明真实 27B 配置、teacher 对拍、20 请求轨迹契约、三臂收益和服务健康仍待上机。随机小模型上常规续训和首拒续训的平均接受数接近，不能据此推出真实 Qwen3.8-27B 的加速收益。

来源：[SpecForge 已提交的适配记录](https://github.com/fg11991/specforge/blob/a99ab23db13f3831e63959d71211b4b2bf9d7249/adaptation_docs/2026-11-10_DSpark%E9%A6%96%E6%8B%92%E4%BD%8D%E7%BD%AE%E8%92%B8%E9%A6%8F.md)。

### 11.3 本文不据此声称的结果

本次能确认“NAIE 基线上的四提交移植和日志入口修复在源码中存在，配套 mock 测试通过”。读取远端日志的连接未成功，因此未确认此版本在 910b3 容器上的实际 trace 输出、TP=4 宽度回传、27B teacher 数值一致性、显存、吞吐或真实蒸馏收益。

完成这些验收后，应补充原始日志路径、请求/回放文件摘要、部署指纹、三臂公平比较和恢复服务健康记录；论文中的加速数字不能填写成我们这套 Ascend 适配的实测结果。

## 12. 参考资料

- [Draft-OPD arXiv v3](https://arxiv.org/html/2605.29343v3)：target-assisted rollout、anchor replay 与 acceptance-aware KL 的原始方法。
- [EMNLP 2026 正式论文](https://aclanthology.org/2026.emnlp-main.1139/)：正式发表版本与元数据。
- [vLLM-Ascend 固定版本](https://github.com/fg11991/vllm-ascend/tree/ce9ca72d19c43bf6c6fce02f6f5dc2fc3958371d)：本文全部适配说明的代码依据。
- [NAIE 基线](https://github.com/fg11991/vllm-ascend/tree/7f93f07236def39e26b491798c7c461a552c2b23)：用于隔离已有功能与本次 OPD 增量。
- [SpecForge 固定版本](https://github.com/fg11991/specforge/tree/a99ab23db13f3831e63959d71211b4b2bf9d7249)：采集、对拍、hidden capture、首拒蒸馏与评测工具。
