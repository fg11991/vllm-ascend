# SpecForge 侧：调用 vLLM-Ascend 抓取服务，生成 DSpark 离线训练特征

> 这份 prompt 给 SpecForge 仓库侧的会话。serving 侧（vLLM-Ascend）的 aux hidden 接口已经实现并跑起来了，现在需要你在 SpecForge 侧生成 token id、调用服务、把落盘产物转成训练格式。
>
> 回复用中文。每一步先给结论和实测数字，再给细节。不确定的地方先问我，不要猜。

## 分工

| 侧 | 负责 | 状态 |
|---|---|---|
| vLLM-Ascend | aux hidden 接口、抓取服务、落盘 | **已完成，服务在跑** |
| **SpecForge（你）** | **token id 渲染、调用服务、转成训练格式、loss_mask** | 待做 |

serving 侧的实现在 `https://github.com/fg11991/vllm-ascend` 分支 `qwen38-capture`（commit `aa7f2f761` 或更新）。它的适配记录、已修的 7 个 bug、操作步骤都在那个分支的 `my_docs/` 下，主要是：

- `my_docs/2026-10-13_hidden抓取_分步操作指南.md` —— 接口契约、落盘格式、故障速查
- `my_docs/2026-10-12_Qwen3.8-Flash-Next_hidden抓取适配与debug.md` —— 为什么是这个口径（含一个会静默掉点的索引陷阱）
- `my_docs/scripts/capture_dataset.py` —— 批量抓取的参考实现，可直接用也可照抄

## 服务端接口契约（已实测）

服务跑在 A3 容器内，`http://127.0.0.1:9811`，`served-model-name` 是 `qwen3.8-flash-next`。**先确认你的代码和服务在不在同一个容器**；不在的话用宿主地址，端口已映射。连通性：

```bash
curl -s http://127.0.0.1:9811/v1/models | python3 -m json.tool | head
```

### 请求：直接发 token id，服务端不渲染

```python
response = requests.post(
    "http://127.0.0.1:9811/v1/completions",
    json={
        "model": "qwen3.8-flash-next",
        "prompt": token_ids,          # list[int]，不是字符串
        "max_tokens": 1,
        "return_token_ids": True,
        "temperature": 0.0,
    },
)
payload = response.json()
assert list(payload["choices"][0]["prompt_token_ids"]) == token_ids   # 逐位校验
path = payload["kv_transfer_params"]["hidden_states_path"]
```

**为什么直接发 token id**：loss mask 由你按 SpecForge 自己的渲染算，token 必须一一对应。让服务端再渲染一次 chat template 就无法保证一致。

### 读回：走 connector 的 flock，不要轮询

```python
from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import load_hidden_states
data = load_hidden_states(path)    # 阻塞直到异步写盘完成
```

没有 vLLM 环境时等价实现：对 `path + ".lock"` 取 `fcntl.LOCK_SH`，然后 `safetensors.torch.load_file(path)`。

### 落盘格式（实测）

| 项 | 值 |
|---|---|
| 文件 | `{shared_storage_path}/{req_id}.safetensors` + 伴随 `.lock` |
| 张量 `hidden_states` | `[T, 6, 2560]`，`bfloat16` |
| 张量 `token_ids` | `[T]`，int64 |
| 第二维 6 个 id 的顺序 | **升序** `[4, 16, 24, 36, 44, 48]` |

**第二维怎么对应到训练侧**：

- 前 5 个（aux id 4/16/24/36/44）= SpecForge 的 tap `[3, 15, 23, 35, 43]`，即 `layers[i+1].attn_hyper_connection(streams)[0]`
- 第 6 个（aux id 48）= `hyper_connection_mixer` 输出 = `last_hidden_state`（这个模型**没有** final norm）

升序由 `EagleModelMixin._set_aux_hidden_state_layers` 的 `tuple(sorted(set(layers)))` 保证，与 SpecForge 的「按 `target_layer_ids` 升序拼接」一致。

### 当前限制

| 限制 | 值 | 原因 |
|---|---|---|
| 样本最长 | **2304 token** | 服务 `--max-model-len 2304`；放宽需要一个默认关闭的开关，且要先做逐位对比验证 |
| 并发 | **≤ 4** | 服务 `--max-num-seqs 4` |
| 吞吐 | 纯 prefill，`--enforce-eager` | decode 速度不相关（`max_tokens=1`） |

## 你要做的三件事

### 1. 生成 token id

用 SpecForge 的渲染口径，这是**硬要求**：开思考、`reasoning_effort=xhigh`、保留历史 reasoning。依据是 `tests/test_data/test_qwen38_template_serving.py`（先读它的 docstring）：训练渲染与 serving 渲染只在这个配置下才逐 token 一致。

**一个待确认项**：`chat_template.jinja` 里控制「保留历史 reasoning」的变量名我们还没拿到。serving 侧不需要它（直接发 token id），但你生成 token id 时需要。请先跑：

```bash
grep -o "preserve[a-z_]*\|reasoning_effort\|enable_thinking\|thinking[a-z_]*" /opt/model/Qwen3.8-Flash-Next-w8a8-mtp/chat_template.jinja | sort -u
```

然后确认 `apply_chat_template` 的实参与它一致。`resolve_chat_template_kwargs` 会**静默丢弃**模板不认识的变量 —— 名字写错不报错，只是不生效。

### 2. 调用服务并落盘

两种方式，选一个：

**方式 A：复用现成脚本（最省事）**

生成一个 JSONL，每行一条：

```json
{"id": "sample-0001", "input_ids": [151644, 872, ...], "loss_mask": [0, 0, 1, ...]}
```

`input_ids` 必需，`id` 可选（决定落盘文件名），**其他字段原样带进 manifest**。然后：

```bash
python3 my_docs/scripts/capture_dataset.py --input samples.jsonl --out-dir /opt/w00958190/capture_dataset --concurrency 4
```

产出 `captures/<id>.safetensors` 和 `manifest.jsonl`（含 id、file、num_tokens、shape 及你带入的字段）。它逐条校验 `token_ids` 一致、`T` 正确、无 NaN/Inf，支持断点续抓。

**方式 B：SpecForge 侧直接调用，一步到位写 `.ckpt`**

跳过中间的 JSONL 和 manifest，在你的脚本里直接「渲染 → 请求 → 读回 → 转换 → 写 `.ckpt`」。请求形状照上面的契约，或参考 speculators 的 `src/speculators/data_generation/vllm_client.py` 的 `generate_hidden_states` / `extract_output`（`/Users/wuyidong/projects/github_projects/job/speculators/speculators`）。

方式 B 更直接，但**必须自己做那三项校验**（`prompt_token_ids` 逐位一致、`T` 等于 prompt 长度、无 NaN/Inf）。

### 3. 转成训练格式

目标格式见 `specforge/algorithms/dspark/providers.py` 的 `OfflineCaptureLayout`：每条样本一个 `.ckpt`（`torch.save` 的 dict），字段 `input_ids [T]`、`loss_mask [T]`、`hidden_states [1, T, 12800]`、`target_last_hidden_states [1, T, 2560]`，**四者的 T 必须相等**。

转换映射：

```python
captured = load_hidden_states(path)["hidden_states"]      # [T, 6, 2560] bf16
taps = captured[:, :5, :]                                  # [T, 5, 2560]
hidden_states = taps.reshape(1, T, 5 * 2560)               # [1, T, 12800]
target_last_hidden_states = captured[:, 5, :].unsqueeze(0) # [1, T, 2560]
```

## ⚠️ 一个会静默掉点的陷阱

**绝不要把 `[T, 6, 2560]` 原样喂给训练读取端。** `specforge/algorithms/common/dflash_family_data.py` 的 `_normalize_hidden_states` 遇到 4 维 hidden 会**静默按流求平均**（那是 DSV4 的口径），不报错，训练照常跑，接受率被悄悄吃掉 —— 与 DeepSeek-V4 锚点偏移同一类失败模式。必须先 reshape 成上面的 2 个张量。

相关的另一个陷阱（serving 侧已处理，供你理解口径）：Transformers 的 `attn_hyper_connection(...)` 在 `[0]` 返回单流门控读出，而 vLLM 的 `GatedResidual.mix()` 在 `[0]` 返回的是 4 路残差原样透传、`[1]` 才是读出。serving 侧取的是 `[1]`，所以落盘的是正确的单流 2560 维。

## 必读代码

| # | 文件 | 看什么 |
|---|---|---|
| 1 | `specforge/offline_capture/qwen4_exp.py` 的 `hyper_connection_taps` | 特征的权威定义：tap i 存 `layers[i+1].attn_hyper_connection(streams)[0]`。**确认 serving 侧落盘的前 5 个与它逐位同义** |
| 2 | `tests/test_offline_capture/test_hf_qwen4_exp.py` 的 `TestTaps` | 用断言写死的口径（输入 `4*hidden`、输出单流 `hidden`）。可照它写转换后的校验 |
| 3 | `specforge/algorithms/dspark/providers.py` 的 `OfflineCaptureLayout` | 训练读取的 `.ckpt` 字段与形状 |
| 4 | `specforge/algorithms/common/dflash_family_data.py` 的 `normalize_dspark_offline_sample` / `_normalize_hidden_states` | **上面那个静默平均的陷阱** |
| 5 | `scripts/prepare_hidden_states.py`（`--help` 与输出目录结构）、`scripts/README.md` 第 1 节 | 现有落盘目录长什么样（`rows_*/`、`dataset_manifest.json`），loss_mask 和 token id 原本怎么生成 |
| 6 | `tests/test_data/test_qwen38_template_serving.py`（先读 docstring） | 渲染口径：只在开思考 + `xhigh` + 保留历史 reasoning 时训练与 serving 逐 token 一致 |
| 7 | speculators 的 `src/speculators/data_generation/vllm_client.py` | 请求与响应解析的参考实现（`generate_hidden_states`、`extract_output`） |
| 8 | vllm-ascend 分支 `qwen38-capture` 的 `my_docs/2026-10-13_hidden抓取_分步操作指南.md` | 接口契约、落盘格式、故障速查 |

## 验收

1. **token 对应**：每个 `.ckpt` 的 `input_ids` 与发给服务的 token id 完全一致，`T` 等于 prompt 长度，`loss_mask` 长度也是 T。
2. **形状与 dtype**：`hidden_states [1, T, 12800]`、`target_last_hidden_states [1, T, 2560]`，四个字段 T 相等。
3. **拼接顺序**：前 5 个 tap 在 12800 维里的顺序是 `[3, 15, 23, 35, 43]` 升序。建议写个断言：取落盘的 `[:, k, :]` 与 `hidden_states` 里第 k 段逐位比较。
4. **不要有静默平均**：转换后的 hidden 是 3 维，确认 `_normalize_hidden_states` 不会走到求平均那条分支。
5. **抓取确定性**：同一条样本抓两次，`hidden_states` 逐位相同（serving 侧已验证过一次，你这边再确认一次）。

## 约束

- 不改训练语义。发现 serving 侧与训练侧不一致时，**给出最小复现（输入、期望、实际）**，由 vLLM-Ascend 侧修，不要在训练侧将就。
- 每次改动记录 commit 和复现命令。没有确认不要 push，不要开 PR。
- 最后在 `my_docs/` 写一份中文记录：渲染口径的确认结果、转换映射、验收数字、遗留问题。

## 需要我这边配合的

1. `chat_template.jinja` 的变量名确认后告诉我，我更新 serving 侧文档。
2. 如果需要样本超过 2304 token，告诉我，serving 侧有个默认关闭的开关可放宽到约 7900 token，但需要先做改前改后的逐位对比验证。
3. aux id 48 对比 `prompt_logprobs` 的交叉验证需要停服务换配置，需要时协调。
