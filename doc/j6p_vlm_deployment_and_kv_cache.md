# J6P VLM 部署与 KV Cache 设计

本文以当前分支已有的 Qwen3.5 状态适配为基础，说明在 Horizon J6P 上部署
一个固定 profile 的 VLM 时，Python 编排层应如何调用两个完整 HBM 图：一个
prefill 图和一个 decode 图，并在两次模型调用之间传递显式 cache state。

本文提供的是部署接口设计和 Python 编排示例。当前分支的
`doc/specs/board_target_handoff_v1.md` 已明确说明：仓库目前没有经过验证的
Horizon BPU Session provider 或 J6P board driver。因此，本文不把 H20 的
Qwen native Session 结果写成 J6P 结果，也不伪造 J6P HBM 的 `infer()` 实现。

示例文件：

`vlaforge/examples/j6p_vlm_kv_cache.py`

测试文件：

`vlaforge/tests/unit/test_j6p_vlm_kv_cache_example.py`

## 1. 部署对象和阶段

推荐先用 Qwen3.5-0.8B 作为 J6P 首个 VLM profile。部署图只有两个可独立
编译和审计的完整 HBM 模型：

```text
image + text tensors
        |
        v
complete prefill HBM
        |
        +--> first_token
        +--> rope_deltas
        +--> explicit state S_0
                     |
                     v
          complete one-token decode HBM
                     |
                     +--> token_(t+1)
                     +--> explicit state S_(t+1)
```

`prefill` 只对当前图像和 prompt 执行一次。之后每生成一个 token，Python
只推进 `step` 并再次调用完整的 decode HBM。J6P provider 应该让 BPU 负责
每个完整图的模型计算、状态读写和必要的 token 选择；Python 只负责输入
契约、生命周期、错误回滚和上层业务循环。

这里的“两个模型”不是两个 layer 子图，也不是若干个自定义 kernel 的集合：

- prefill HBM 内含视觉编码、图像 token 注入、文本 embedding、完整语言层
  prefill、RoPE、attention/linear-attention、MLP、norm 和输出 head；
- decode HBM 内含单 token embedding、完整语言层 decode、显式 cache 更新、
  RoPE、attention/linear-attention、MLP、norm 和输出 head；
- 两个 HBM 都必须是 J6P 编译器可接受的完整标准算子图，不能依赖 custom op；
- Python 不调用任何 layer，也不在 Python 中实现 kernel。层级信息只用于生成
  和校验 state manifest，不能成为运行时调度单元。

当前 Qwen3.5 适配位于
`vlaforge/python/vlaforge/adapters/qwen3_5/qwen3_5_state.py`，已经把
upstream 的混合状态拆成显式 prefill/decode 接口。它没有把所有状态都称作
KV，因为 Qwen3.5 同时包含 full-attention 和 linear-attention 层。

## 2. 两个完整 HBM 的 I/O 差异

prefill 和 decode 的参数不应共用一个模糊的字典。应分别保存
`prefill_manifest.json` 和 `decode_manifest.json`，每个 manifest 至少包含
完整 HBM 的 hash、输入/输出顺序、shape、dtype、layout、state schema hash、
compiler/runtime 版本和 custom-op 审计结果。

| 项目 | 完整 prefill HBM | 完整 decode HBM |
|---|---|---|
| 输入 | `input_ids`、`attention_mask`、`pixel_values`、`image_grid_thw`、`mm_token_type_ids` | 当前 token、`step`/`cache_position`、`rope_deltas`、完整显式 state |
| 序列形态 | `[B, prompt_length]`，固定 profile | `[B, 1]`，每次一个 token |
| 视觉路径 | 包含 vision encoder 和 image token 注入 | 不重新执行 vision encoder |
| 输出 | first token、`rope_deltas`、prefill 后 state | next token/logits、更新后的 state |
| state 角色 | 产生 `S_prefill` | 消费并产生 `S_decode` |
| 算子集合 | vision + prefill language graph | decode language graph + state update |
| 自定义算子 | 必须为 0 | 必须为 0 |

prefill 输出 state 与 decode 输入 state 必须通过一个明确的 typed ABI 对齐。
如果两份导出使用不同的物理布局，不能在 Python 里静默 reshape 或按 layer
拼接；要么在导出时统一 ABI，要么由 provider 明确实现并测量 host-side
repack/copy。不能因此引入 J6P 不支持的自定义算子，也不能把 repack 时间
隐藏在 decode latency 中。

## 3. Cache state 的真实组成

每个 Transformer layer 产生两组状态，顺序由模型配置中的 `layer_types`
固定。这个信息只用于生成两个完整 HBM 的 state ABI 和做 manifest 校验，
不是让 J6P 在运行时逐 layer 调度。provider manifest 必须保存顺序、shape、
dtype 和总数量；不能在 Python 侧依赖“第几个 tensor 看起来像 key”来推断。

| Layer type | State | 典型布局 | 生命周期 |
|---|---|---|---|
| `full_attention` | key | `[B, H_kv, max_seq, D]` | prefill 写入 prompt，decode 按 `cache_position` 写入一个 token |
| `full_attention` | value | `[B, H_kv, max_seq, D]` | 与 key 同步更新 |
| `linear_attention` | convolution state | `[B, 3 * H_k * D_k, K]` | prefill 初始化，decode 每 token 滚动更新 |
| `linear_attention` | recurrent state | `[B, H_v, D_k, D_v]` | prefill 初始化，decode 每 token 更新 |
| model control | `rope_deltas` | profile-specific tensor | prefill 生成，decode 只读 |
| runtime control | `step`, `cache_position` | `int64` scalar/tensor | Python 或 compiled loop 推进，不属于模型 cache tensor |

full-attention 的物理容量应在编译时固定为 `max_sequence_length`。有效长度
由 `prompt_length + generated_tokens - 1` 控制；未使用槽位必须由 attention
mask 屏蔽。decode 的 `cache_position` 是逻辑 token 位置，不是“当前数组中
第几个有效元素”。如果以后引入分页或压缩 cache，也必须同时维护逻辑位置
和物理地址，不能只缩短 tensor 后继续使用旧的 RoPE 位置。

对于 Qwen3.5，`step=1` 对应把第一个生成 token 写入
`cache_position=prompt_length`。这是当前
`Qwen3_5ExplicitDecodeStep` 的约定，示例也遵循这个约定。

## 4. Cache key 和失效规则

cache 只能在输入语义完全相同时复用。推荐 cache key 至少包含：

```text
model_id + checkpoint/artifact hash + profile
+ input_ids + attention_mask + pixel_values + image_grid_thw
+ mm_token_type_ids + prompt_length + dtype/layout
```

下列任一项变化都必须重新 prefill：

- 图像内容、图像尺寸或 `image_grid_thw` 变化；
- prompt token、padding mask 或多模态 token type 变化；
- checkpoint、prefill/decode HBM、shape profile、dtype 或 layout 变化；
- Session reset、用户切换、失败恢复或 batch slot 重新分配。

因此不能把“同一张图像”简单理解为可以跨 prompt 复用完整 prefill，也不能
把视觉 token 的混合 hidden state 当成一个只依赖文本的静态 cache。示例中的
`input_identity()` 对所有 prefill 输入计算 SHA256；真实 provider 还应把
artifact manifest 的 SHA256 纳入身份。

## 5. Session 生命周期和事务语义

一次请求应按以下顺序执行：

1. 创建或取得一个独占 Session，并确认 provider、HBM、runtime 和 target descriptor。
2. 校验 batch、prompt 长度、生成长度、dtype、layout 和 state schema。
3. `reset()` 清除旧的 device state 和上一次未提交的 output。
4. 调用 `prefill()`，得到 first token、`rope_deltas` 和 `S_0`。
5. 以 `step=1` 开始 decode；每次 decode 成功后才替换 Python 侧 state 引用。
6. 完成固定长度生成后提交 output identity；失败时清除整个 Session state。

不要在多个 Session 之间共享可写 cache tensor。多路并发请求应使用独立
Session 或明确的 cache slot/ownership 协议；仅复制 Python tuple 不会复制
device memory。

当前示例把 `reset()` 放在每个新 image+prompt 请求前，并在异常路径再次调用。
这是一种保守的 single-request 模式。要做连续对话，需要额外定义：哪些文本
token 可以保留、图像是否保持不变、最大 context 如何增长，以及 reset 后哪些
state 必须重新 materialize；不能通过关闭 reset 来“复用 KV”。

## 6. J6P provider 的窄接口

示例定义了 `J6PCompleteHbmProvider` 这个 seam：

```python
class J6PCompleteHbmProvider(Protocol):
    model_id: str
    max_sequence_length: int
    prefill_manifest: CompleteHbmManifest
    decode_manifest: CompleteHbmManifest

    def reset(self) -> None: ...
    def prefill(self, *, input_ids, attention_mask, pixel_values,
                image_grid_thw, mm_token_type_ids) -> PrefillResult: ...
    def decode(self, *, token, step, rope_deltas, state) -> DecodeResult: ...
```

这个接口有意比模型内部实现小，并且只暴露两个完整 HBM 的调用：

- provider 内部负责加载完整 prefill/decode 两个 HBM、绑定各自不同的输入
  输出、分配稳定的 cache buffer、调用 HBRT/UCP，并检查实际设备和 ABI；
- Python 负责 profile 和输入契约、cache key、请求生命周期、错误回滚；
- model adapter 负责把 upstream Qwen3.5 的状态顺序、RoPE 和视觉输入转换成
  两份 provider manifest，不把 Qwen 特殊分支塞入公共 runtime。

provider 的 `prefill()` 和 `decode()` 都对应一次完整 HBM 模型调用。不能把
`decode()` 实现成“调用第 7 层、第 8 层……”的 Python 循环，也不能用
custom op 把未支持的 layer 偷渡进 BPU。

一个合格的 J6P provider 在 `prefill()` 和 `decode()` 返回前还应记录：
  HBM hash、输入输出 shape/dtype、state schema hash、执行设备、CPU fallback、
  BPU completion boundary 和错误码。`reset()` 必须能证明旧 state 不会被下一
  个请求读到。

## 7. Python 示例怎么接真实 HBM

示例中的 `J6PCompleteHbmProviderImpl` 只是一个明确的待接入适配器。真实
接入时，建议按下面顺序实现，而不是在 Python 中把所有 KV 拼成一个大字典：

```python
prefill_manifest = CompleteHbmManifest.from_json(
    Path("/data/models/qwen35_08b/prefill_manifest.json")
)
decode_manifest = CompleteHbmManifest.from_json(
    Path("/data/models/qwen35_08b/decode_manifest.json")
)
provider = load_j6p_qwen35_08b_provider(
    prefill_manifest=prefill_manifest,
    decode_manifest=decode_manifest,
)
deployment = J6PVLMDeployment(provider)
result = deployment.generate(prepared, new_tokens=16)
```

`prepared` 应由 processor/预处理模块生成并固定成 target profile，至少包括：

- `input_ids`、`attention_mask`；
- `pixel_values` 和 `image_grid_thw`；
- `mm_token_type_ids`；
- `prompt_length`。

J6P 上的图像 resize、patch layout、归一化和 dtype 转换必须在输入契约中明确
记录。不能直接拿 H20 的 CUDA tensor、TorchScript bundle 或 H20 native C++
library 当作 J6P provider。

## 8. 内存和性能边界

建议把内存拆成四项记录：

1. HBM 静态 plan：权重、常量、workspace、输入/输出和 state buffer；
2. provider 分配的 device/cache memory；
3. Python 进程 RSS；
4. host-to-device、device-to-host 和 BPU completion 时间。

不能把进程 RSS 与 HBM plan 相加作为设备总内存，也不能用 Python 文件链墙钟
时间冒充每 token latency。正式测量至少应分别记录：prefill latency、decode
per-token latency、固定生成长度总延迟、tokens/s、cache bytes、峰值内存和
CDF。若 provider 含 CPU fallback，必须按 stage 记录并单独标识。

## 9. 例外和失败处理

| Failure | Required action |
|---|---|
| state 数量或 schema hash 不匹配 | 在首次 prefill 前拒绝运行 |
| prompt + generation 超出静态容量 | 在 provider 调用前拒绝 |
| image/prompt/profile 改变 | reset 后重新 prefill |
| decode 失败或输出不完整 | 丢弃整次请求 state，不复用部分 cache |
| BPU ownership 或 runtime ABI 不匹配 | 不启动模型，保留 preflight 证据 |
| 仅有 x86/H20 artifact，没有 J6P HBM | 标记 pending，不能宣称 board deployment |

## 10. 当前实现边界

当前分支已经有 Qwen3.5 的显式状态拆分和 H20 固定 profile 的 native
deployment evidence，但还缺少：

- J6P 的实际 BPU provider 和 board driver；
- Qwen3.5-0.8B 的两个完整 J6P HBM（prefill 和 decode）及其 manifests；
- J6P 上的 state layout、CPU fallback 和 runtime ABI 实测；
- J6P prefill/decode CDF、内存和完整输出核验。

所以本文和示例完成的是两个完整 HBM 的 Python 编排接口与 cache 设计，不能作为 J6P VLM
已经部署成功的证明。J6P 实验应在 provider、HBM、target descriptor 和输入
manifest 齐全后重新执行。

## References

- [VLAForge board target handoff](specs/board_target_handoff_v1.md)
- [Qwen3.5 explicit state adapter](../vlaforge/python/vlaforge/adapters/qwen3_5/qwen3_5_state.py)
- [Qwen3.5 native H20 report](reports/qwen35_native_formal_20260910.md)
- [Hugging Face cache explanation](https://huggingface.co/docs/transformers/main/en/cache_explanation)
- [Horizon runtime development guide](https://doc.oe.horizon.auto/en/guide/model_deployment/board_deployment/runtime_dev.html)
