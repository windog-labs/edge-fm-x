# SmolVLA Phase-1 Horizon 使用说明

本文说明 EdgeFM 当前如何支持 SmolVLA 的 LLM 部分，也就是：

- `prefill`: 输入 prefix embedding/mask/position，输出每层 prefix KV cache。
- `decode`: 输入 suffix embedding/mask/position，并使用 prefix KV cache，输出 `expert_hidden`。

当前范围不包含 ViT、`embed_suffix` 和 `action_out_proj`。这些模块应在 EdgeFM 外部准备输入或消费输出。

重要前提：

- 这里说的 `smolvla` 指的是 **LeRobot 的 `SmolVLAPolicy`**，其中包含 `vlm_with_expert`，也就是
  - VLM prefix 16 层
  - Action Expert 16 层
- 它**不是**把一个单独的 `SmolVLM2-500M-Video-Instruct` 文本/多模态 backbone 生硬拆成 `prefill` 和 `decode` 两个阶段。
- `SmolVLM2-500M-Video-Instruct` 这类 standalone SmolVLM2 checkpoint 只有 VLM 主干，没有 SmolVLA 的 `action_expert`、`embed_suffix` 和 `action_out_proj`，因此不能直接作为本文件所述 phase-1 双 stage 导出的完整来源模型。

## 已确认的 LeRobot 版本与模型

当前应以 LeRobot policy checkpoint 为入口：

```text
policy repo: lerobot/smolvla_base
repo sha: c83c3163b8ca9b7e67c509fffd9121e66cb96205
VLM backbone: HuggingFaceTB/SmolVLM2-500M-Video-Instruct
load_vlm_weights: true
num_vlm_layers: 16
num_expert_layers: 0
attention_mode: cross_attn
self_attn_every_n_layers: 2
expert_width_multiplier: 0.75
chunk_size: 50
max_action_dim: 32
```

注意：

- `lerobot/smolvla_base` 的 `model.safetensors` 包含 `SmolVLAPolicy` 权重，因此它才是 phase-1 导出需要的 checkpoint。
- `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` 只是该 policy config 中的 VLM backbone 名称，不能单独替代 `lerobot/smolvla_base`。
- 当前 `horizon_quant` 使用 Python 3.10。LeRobot main / v0.5.x 已要求 Python 3.12，不能直接作为 Horizon 编译源码入口。建议使用 LeRobot `v0.4.4` checkout，它仍是 Python 3.10 兼容，并且包含 `SmolVLAPolicy/vlm_with_expert`。

本地建议准备：

```bash
git -C ~/Repos/public/lerobot worktree add ~/Repos/public/lerobot-v0.4.4 v0.4.4

source ~/miniconda3/bin/activate horizon_quant
pip install 'draccus==0.10.0' 'einops>=0.8.0,<0.9.0' \
  'gymnasium>=1.1.1,<2.0.0' 'diffusers>=0.27.2,<0.36.0'
```

如果直连 Hugging Face 超时，可以使用镜像下载 policy checkpoint：

```bash
HF_ENDPOINT=https://hf-mirror.com python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="lerobot/smolvla_base",
    repo_type="model",
    endpoint="https://hf-mirror.com",
    local_dir="examples/smolvla/SmolVLA-Base",
    allow_patterns=[
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ],
)
PY
```

`examples/smolvla/SmolVLA-Base` 已被仓库 `.gitignore` 覆盖，不应提交模型权重。

如果本地已经有 standalone VLM backbone，例如 `examples/smolvla/SmolVLM2-500M-Video-Instruct`，导出时可以避免再次从 Hub 下载 backbone：

```bash
export EDGE_FM_SMOLVLA_VLM_MODEL_PATH=$PWD/examples/smolvla/SmolVLM2-500M-Video-Instruct
export EDGE_FM_SMOLVLA_DEVICE=cpu
export EDGE_FM_LEROBOT_ROOT=/home/zhangzimo/Repos/public/lerobot-v0.4.4
```

`EDGE_FM_SMOLVLA_VLM_MODEL_PATH` 只覆盖 policy config 中的 `vlm_model_name`，不会改变 `lerobot/smolvla_base` 的 policy/action expert 权重加载。

## 接口原则

SmolVLA 不使用模型专用的 `smolvla_prefill` 或 `smolvla_decode` public API。它复用 EdgeFM 通用 tensor stage 接口：

```cpp
TensorMap EdgeFM::prefill(int32_t request_id, const TensorRefMap& inputs) const;
TensorMap EdgeFM::decode(int32_t request_id, const TensorRefMap& inputs) const;
```

Python 绑定对应为：

```python
prefill_outputs = engine.prefill(request_id, inputs)
decode_outputs = engine.decode(request_id, inputs)
```

`generate()` 仍然是 token generation API，主要服务 Qwen LLM/VLM 的采样式生成。SmolVLA phase-1 的 denoise expert 是 tensor-in/tensor-out forward，因此走 `decode()` stage，而不是 `generate()`。

## 导出两个 Horizon 模型

SmolVLA phase-1 需要导出两个 HBM：

- `smolvla_prefill.hbm`: 负责 prefix prefill，输出 `prefix_kv_layer_*`。
- `smolvla_decode.hbm`: 负责 suffix/action expert decode，输入 `suffix_*` 和 `prefix_kv_layer_*`，输出 `expert_hidden`。

第一步，准备 engine config。关键字段示例：

```json
{
  "model_name": "smolvla",
  "runtime": {
    "device": "horizon",
    "hw_profile": "j6m"
  },
  "prefill_model_path": "examples/smolvla/SmolVLA-Base",
  "kvcache": {
    "dtype": "fp16",
    "attention_type": "gqa",
    "requests": [
      {
        "request_id": 0,
        "prefix_token_ids": [],
        "max_tokens": 128
      }
    ]
  },
  "smolvla": {
    "prefix_len": 128,
    "suffix_len": 50,
    "num_layers": 16,
    "lerobot_root": "/home/zhangzimo/Repos/public/lerobot-v0.4.4"
  }
}
```

第二步，生成 Horizon compile spec：

```bash
python - <<'PY'
import edge_fm

engine = edge_fm.EdgeFM("smolvla_horizon_engine.json")
engine.tune()
PY
```

`tune()` 会写出 `compile_spec.json`，并在日志中打印：

```text
Horizon compile spec written: /path/to/compile_spec.json
```

如果需要手动查找最近的 spec，也可以用：

```bash
find ~/.cache/edge-fm/backend_artifacts -name compile_spec.json -printf '%T@ %p\n' \
  | sort -n \
  | tail -1
```

第三步，分别编译 `prefill` 和 `decode` stage。EdgeFM helper 可以直接完成：

- generated module 初始化
- ONNX export
- J6M rewrite 诊断文件生成
- `hb_compile` YAML 生成
- `hb_compile` 调用
- 将 HBM 复制到 `compile_spec.json` 中对应 stage 的 `artifact_path`

```bash
SPEC=/path/to/compile_spec.json

python scripts/horizon/compile_horizon_from_spec.py "$SPEC" \
  --stage prefill \
  --horizon-rewrite auto \
  --export-onnx \
  --hb-compile

python scripts/horizon/compile_horizon_from_spec.py "$SPEC" \
  --stage decode \
  --horizon-rewrite auto \
  --export-onnx \
  --hb-compile
```

如果已经导出过 ONNX，不想重复 trace，可复用现有 ONNX：

```bash
python scripts/horizon/compile_horizon_from_spec.py "$SPEC" \
  --stage prefill \
  --horizon-rewrite on \
  --skip-model-init \
  --hb-compile \
  --reuse-onnx \
  --onnx-path "$PWD/.tmp_codex/smolvla_phase1/smolvla_prefill.onnx"
```

不传 `--compiler-command` 或加 `--dry-run` 时，helper 不会调用外部编译器，只会生成 preparation manifest，便于检查 stage I/O：

```bash
python scripts/horizon/compile_horizon_from_spec.py "$SPEC" --stage prefill --dry-run
python scripts/horizon/compile_horizon_from_spec.py "$SPEC" --stage decode --dry-run
```

`--stage` 只接受 `prefill` 和 `decode`。旧的 `expert_denoise` 名称不再作为 public stage 名称。

Horizon 编译默认参数：

```text
march: nash-m
onnx opset: 17
onnx IR version: clamped to <= 9
input_type_rt/input_type_train: featuremap
norm_type: no_preprocess
compile_mode: latency
optimize_level: O0
core_num: 1
```

SmolVLA generated adapter 默认使用 `EDGE_FM_SMOLVLA_EXPORT_DTYPE=float32`，会在导出前把 LeRobot policy 转成 FP32。这个默认值是必要的：保留原生 BF16 时，decode stage 会在 HMCT shape inference 中触发 `FLOAT vs BFLOAT16` 类型冲突。若只做 PyTorch 诊断，可以设置 `EDGE_FM_SMOLVLA_EXPORT_DTYPE=keep` 保留模型原始 dtype。

## Stage I/O

实际 shape 和 dtype 以 `compile_spec.json` 中的 `stages` 为准。默认生成规则如下。

`prefill` inputs:

```text
prefix_embeds: [1, prefix_len, hidden_size]
prefix_attention_mask: [1, prefix_len, prefix_len] uint8, 0/1 mask
prefix_position_ids: [1, prefix_len]
```

`prefill` outputs:

```text
prefix_kv_layer_0: [2, prefix_len, num_kv_heads, head_dim]
...
prefix_kv_layer_N: [2, prefix_len, num_kv_heads, head_dim]
```

`decode` inputs:

```text
suffix_embeds: [1, suffix_len, expert_hidden_size]
denoise_attention_mask: [1, suffix_len, prefix_len + suffix_len] uint8, 0/1 mask
suffix_position_ids: [1, suffix_len]
prefix_kv_layer_0: [2, prefix_len, num_kv_heads, head_dim]
...
prefix_kv_layer_N: [2, prefix_len, num_kv_heads, head_dim]
```

`decode` outputs:

```text
expert_hidden: [1, suffix_len, expert_hidden_size]
```

当前 Horizon tensor stage runtime 期望输入 `Tensor` 位于 CPU，并且 shape/dtype 与 HBM runtime I/O 完全一致。

mask 输入使用 `uint8` 的 0/1 表示，generated adapter 会在调用 LeRobot `vlm_with_expert.forward()` 前转换为 `torch.bool`。这保持了 LeRobot `make_att_2d_masks()` 的 bool mask 语义，同时避免 EdgeFM public Tensor API 额外引入 bool dtype。

KV cache 和 `expert_hidden` 的 public I/O dtype 目前固定为 `float32`。LeRobot 内部会使用 backbone 权重 dtype（当前为 bfloat16），generated adapter 在 stage 边界执行转换：prefill 输出 KV 转 `float32`，decode 输入 KV 转回模型权重 dtype，decode 输出 `expert_hidden` 转 `float32`。这样可以避开当前 Horizon runtime public Tensor path 不支持 BF16 的限制。

## Python 调用示例

下面示例假设 `embed_suffix` 已在 EdgeFM 外部完成，且输入 numpy array 的 shape/dtype 与 `compile_spec.json` 一致。

```python
from __future__ import annotations

import numpy as np
import edge_fm


def tensor_from_numpy(array: np.ndarray) -> edge_fm.Tensor:
    array = np.ascontiguousarray(array)
    if array.dtype == np.float32:
        dtype = edge_fm.DType.Float32
    elif array.dtype == np.float16:
        dtype = edge_fm.DType.Float16
    elif array.dtype == np.int32:
        dtype = edge_fm.DType.Int32
    elif array.dtype == np.uint8:
        dtype = edge_fm.DType.UInt8
    else:
        raise TypeError(f"unsupported dtype: {array.dtype}")

    return edge_fm.Tensor(
        int(array.ctypes.data),
        list(array.shape),
        dtype,
        edge_fm.Device.CPU,
        0,
        True,
    )


engine = edge_fm.EdgeFM("smolvla_horizon_engine.json")
request_id = 0

prefix_embeds_np = np.zeros((1, 128, 960), dtype=np.float32)
prefix_mask_np = np.ones((1, 128, 128), dtype=np.uint8)
prefix_pos_np = np.arange(128, dtype=np.int32).reshape(1, 128)

prefill_outputs = engine.prefill(
    request_id,
    {
        "prefix_embeds": tensor_from_numpy(prefix_embeds_np),
        "prefix_attention_mask": tensor_from_numpy(prefix_mask_np),
        "prefix_position_ids": tensor_from_numpy(prefix_pos_np),
    },
)

suffix_embeds_np = np.zeros((1, 50, 720), dtype=np.float32)
denoise_mask_np = np.ones((1, 50, 178), dtype=np.uint8)
suffix_pos_np = np.arange(128, 178, dtype=np.int32).reshape(1, 50)

decode_outputs = engine.decode(
    request_id,
    {
        "suffix_embeds": tensor_from_numpy(suffix_embeds_np),
        "denoise_attention_mask": tensor_from_numpy(denoise_mask_np),
        "suffix_position_ids": tensor_from_numpy(suffix_pos_np),
    },
)

expert_hidden = decode_outputs["expert_hidden"]
```

示例中 `copy_data=True`，因此返回的 `Tensor` 拥有自己的 CPU buffer。若改成 zero-copy view，调用方必须保证 numpy buffer 活到 `prefill()` 或 `decode()` 返回之后。

同一个 `engine` 和同一个 `request_id` 下，`prefill()` 会把 `prefix_kv_layer_*` 缓存在 EdgeFM 内部，所以 `decode()` 可以只传 suffix 相关输入。

如果需要显式传 KV cache，例如跨进程、跨 engine，或希望把 KV 交给外部生命周期管理，则把 `prefill_outputs` 中的 `prefix_kv_layer_*` 也加入 decode inputs：

```python
decode_inputs = {
    "suffix_embeds": tensor_from_numpy(suffix_embeds_np),
    "denoise_attention_mask": tensor_from_numpy(denoise_mask_np),
    "suffix_position_ids": tensor_from_numpy(suffix_pos_np),
}
decode_inputs.update(
    {
        name: tensor
        for name, tensor in prefill_outputs.items()
        if name.startswith("prefix_kv_layer_")
    }
)

decode_outputs = engine.decode(request_id, decode_inputs)
```

## C++ 调用示例

下面示例只展示 EdgeFM API 结构。真实业务中，`Tensor::view` 或 `Tensor::clone_from` 的 data pointer、shape 和 dtype 应来自你的输入 buffer。

```cpp
#include <edge-fm/edge-fm.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

int main() {
    edge_fm::EdgeFM engine("smolvla_horizon_engine.json");
    const int32_t request_id = 0;

    std::vector<float> prefix_embeds_buffer(1 * 128 * 960);
    std::vector<uint8_t> prefix_mask_buffer(1 * 128 * 128, 1);
    std::vector<int32_t> prefix_pos_buffer(1 * 128);
    std::vector<float> suffix_embeds_buffer(1 * 50 * 720);
    std::vector<uint8_t> denoise_mask_buffer(1 * 50 * 178, 1);
    std::vector<int32_t> suffix_pos_buffer(1 * 50);

    edge_fm::Tensor prefix_embeds = edge_fm::Tensor::view(
        prefix_embeds_buffer.data(),
        {1, 128, 960},
        edge_fm::DType::Float32,
        edge_fm::Device::CPU);
    edge_fm::Tensor prefix_attention_mask = edge_fm::Tensor::view(
        prefix_mask_buffer.data(),
        {1, 128, 128},
        edge_fm::DType::UInt8,
        edge_fm::Device::CPU);
    edge_fm::Tensor prefix_position_ids = edge_fm::Tensor::view(
        prefix_pos_buffer.data(),
        {1, 128},
        edge_fm::DType::Int32,
        edge_fm::Device::CPU);

    edge_fm::TensorRefMap prefill_inputs{
        {"prefix_embeds", &prefix_embeds},
        {"prefix_attention_mask", &prefix_attention_mask},
        {"prefix_position_ids", &prefix_position_ids},
    };
    edge_fm::TensorMap prefill_outputs = engine.prefill(request_id, prefill_inputs);

    edge_fm::Tensor suffix_embeds = edge_fm::Tensor::view(
        suffix_embeds_buffer.data(),
        {1, 50, 720},
        edge_fm::DType::Float32,
        edge_fm::Device::CPU);
    edge_fm::Tensor denoise_attention_mask = edge_fm::Tensor::view(
        denoise_mask_buffer.data(),
        {1, 50, 178},
        edge_fm::DType::UInt8,
        edge_fm::Device::CPU);
    edge_fm::Tensor suffix_position_ids = edge_fm::Tensor::view(
        suffix_pos_buffer.data(),
        {1, 50},
        edge_fm::DType::Int32,
        edge_fm::Device::CPU);

    edge_fm::TensorRefMap decode_inputs{
        {"suffix_embeds", &suffix_embeds},
        {"denoise_attention_mask", &denoise_attention_mask},
        {"suffix_position_ids", &suffix_position_ids},
    };

    edge_fm::TensorMap decode_outputs = engine.decode(request_id, decode_inputs);
    const edge_fm::Tensor& expert_hidden = decode_outputs.at("expert_hidden");
    (void)expert_hidden;

    return 0;
}
```

如果不依赖内部 request cache，也可以显式把 prefill 输出 KV 作为 decode 输入：

```cpp
for (auto& item : prefill_outputs) {
    if (item.first.rfind("prefix_kv_layer_", 0) == 0) {
        decode_inputs[item.first] = &item.second;
    }
}
```

注意：显式传 KV 时，`prefill_outputs` 必须活到 `engine.decode()` 调用结束，因为 `TensorRefMap` 只保存 `Tensor*`。

## J6M 构建与验证状态

以下流程已在 2026-04-27 验证通过。

Host HBM 编译环境：

```text
conda env: horizon_quant
hb_compile: 3.5.3
hmct: 2.5.6
hbdk: 4.5.5
target march: nash-m
LeRobot root: /home/zhangzimo/Repos/public/lerobot-v0.4.4
policy checkpoint: examples/smolvla/SmolVLA-Base
VLM override: examples/smolvla/SmolVLM2-500M-Video-Instruct
```

实测产物：

```text
compile_spec: /home/zhangzimo/.cache/edge-fm/backend_artifacts/c86aba310f8c259c/compile_spec.json
prefill HBM: /home/zhangzimo/.cache/edge-fm/backend_artifacts/c86aba310f8c259c/smolvla_prefill.hbm
decode HBM: /home/zhangzimo/.cache/edge-fm/backend_artifacts/c86aba310f8c259c/smolvla_decode.hbm
prefill ONNX: .tmp_codex/smolvla_phase1/smolvla_prefill.onnx
decode ONNX: .tmp_codex/smolvla_phase1/smolvla_decode.onnx
prefill hb_compile log: .tmp_codex/smolvla_phase1/prefill_hb_compile.log
decode hb_compile log: .tmp_codex/smolvla_phase1/decode_hb_compile.log
```

J6M Docker 交叉构建：

```bash
EDGE_FM_BUILD_JOBS=4 bash scripts/docker/build_hrz.sh install
```

`scripts/docker/build_hrz.sh` 会在本机存在 OpenExplorer v3.5.0 deps 时自动挂载：

```text
host: ~/Packages/horizon_j6_open_explorer_v3.5.0-py310_20250927/samples/ucp_tutorial/deps_aarch64
container: /opt/horizon_deps_aarch64
```

并默认启用 `ENABLE_HORIZON_RUNTIME=ON`。如果 deps 在其他路径，使用：

```bash
EDGE_FM_HOST_HORIZON_DEPS_ROOT=/path/to/deps_aarch64 \
EDGE_FM_BUILD_JOBS=4 \
bash scripts/docker/build_hrz.sh install
```

构建完成后，`build-j6m/install/lib/libedge_fm.so` 是 aarch64 产物，并直接链接 Horizon runtime：

```text
libdnn.so
libhbucp.so
libhbrt4.so
libhbtl.so
libhb_arm_rpc.so
...
```

板端 `j6m-1` 验证：

```bash
scp /home/zhangzimo/.cache/edge-fm/backend_artifacts/c86aba310f8c259c/smolvla_prefill.hbm \
    /home/zhangzimo/.cache/edge-fm/backend_artifacts/c86aba310f8c259c/smolvla_decode.hbm \
    j6m-1:/userdata/edgefm_smolvla_phase1/

ssh j6m-1 "cd /data/apps/hrt_model_exec/script && \
  export LD_LIBRARY_PATH=../aarch64/lib:\$LD_LIBRARY_PATH && \
  ../aarch64/bin/hrt_model_exec model_info \
  --model_file=/userdata/edgefm_smolvla_phase1/smolvla_prefill.hbm"

ssh j6m-1 "cd /data/apps/hrt_model_exec/script && \
  export LD_LIBRARY_PATH=../aarch64/lib:\$LD_LIBRARY_PATH && \
  ../aarch64/bin/hrt_model_exec model_info \
  --model_file=/userdata/edgefm_smolvla_phase1/smolvla_decode.hbm"
```

dummy raw tensor perf smoke：

```text
prefill: frame latency 17.649 ms, 56.449 FPS, frame_count=1, thread_num=1
decode: frame latency 12.125 ms, 81.867 FPS, frame_count=1, thread_num=1
```

EdgeFM C++ API smoke 也已在 `j6m-1` 上验证通过。该 smoke 不走 `hrt_model_exec`，而是直接链接 `libedge_fm.so`，调用 `EdgeFM::prefill()` 和 `EdgeFM::decode()`：

```bash
ssh j6m-1 "cd /userdata/edgefm_smolvla_phase1 && \
  export EDGE_FM_CONFIG_DIR=/userdata/edgefm_smolvla_phase1/config && \
  export LD_LIBRARY_PATH=/userdata/edgefm_smolvla_phase1:/data/apps/hrt_model_exec/aarch64/lib:\$LD_LIBRARY_PATH && \
  ./edge_fm_smolvla_horizon_smoke \
  /userdata/edgefm_smolvla_phase1/smolvla_horizon_engine_board.json"
```

输出确认：

```text
SmolVLA Horizon prefill/decode smoke passed
prefill outputs: 16
expert_hidden shape: [1, 50, 720]
```

这个 smoke 验证了 edge-fm runtime 的两件事：

- `prefill()` 可以加载 `smolvla_prefill.hbm` 并返回 16 个 `prefix_kv_layer_*`。
- `decode()` 可以复用同一个 `request_id` 下 prefill 缓存的 KV，并调用 `smolvla_decode.hbm` 输出 `expert_hidden`。

这只证明 HBM 能在 J6M BPU runtime 上实际执行，不代表端到端 policy 精度。精度验证仍需接入真实 `embed_suffix`、真实 prefix embeddings/mask/position、真实 denoise mask、以及 `action_out_proj`。

## J6M EdgeFM 性能数据

以下性能数据在 2026-04-27 通过 EdgeFM C++ API 在 `j6m-1` 板端实测，不走 `hrt_model_exec`。测试二进制和运行位置为：

```text
host binary: build-j6m/bin/edge_fm_smolvla_horizon_benchmark
board dir: /data/edgefm_smolvla_phase1_benchmark
board host: j6m-1 (hostname: hobot)
runtime libs: /data/apps/hrt_model_exec/aarch64/lib
warmup: 3
iterations: 20
inputs: dummy CPU tensors, shape/dtype 与 HBM I/O 完全一致
```

运行命令模板：

```bash
ssh j6m-1 "cd /data/edgefm_smolvla_phase1_benchmark && \
  export EDGE_FM_CONFIG_DIR=/data/edgefm_smolvla_phase1_benchmark/config && \
  export LD_LIBRARY_PATH=/data/edgefm_smolvla_phase1_benchmark:/data/apps/hrt_model_exec/aarch64/lib:\$LD_LIBRARY_PATH && \
  ./edge_fm_smolvla_horizon_benchmark \
    /data/edgefm_smolvla_phase1_benchmark/p512_s32/smolvla_horizon_engine_board.json \
    --prefix-len=512 --suffix-len=32 --warmup=3 --iterations=20 --stage=both"
```

计时范围：

- `prefill`: 对应 SmolVLA 的 prefix prefill 阶段，也就是 `EdgeFM::prefill()`。输入是 `prefix_embeds / prefix_attention_mask / prefix_position_ids`，输出是 16 层 `prefix_kv_layer_*`。计时包含 CPU 输入 copy、BPU forward、输出 KV copy，以及 EdgeFM 内部 KV cache 更新。
- `decode`: 对应 SmolVLA denoise 中的 LLM / `action_expert` 阶段，也就是 `EdgeFM::decode()`。测试前会先跑一次不计时的 `prefill()` 填充 request cache；decode 计时本身包含 suffix 输入 copy、cached KV 注入 copy、BPU forward、`expert_hidden` 输出 copy。
- 当前测试是 phase-1 stage runtime 性能，不包含 ViT、`embed_suffix`、完整 denoise loop 调度、`action_out_proj` 或真实业务预处理/后处理。

Prefill 阶段：

这里应该只看 `prefix_len`。每个 `prefix_len` 实际都跑了两组配对 case，也就是 `suffix_len=32` 和 `suffix_len=64`。由于 prefill 本身不消费 suffix，下面主表给出两组 case 的代表值，按两次 `mean_ms` 取平均；每组原始 case 的结果仍保留在 `board_benchmark_summary.json` 中。

| prefix_len | paired case means ms (`s32 / s64`) | representative mean ms |
|---:|---:|---:|
| 512 | 74.252 / 74.275 | 74.264 |
| 1024 | 248.250 / 248.293 | 248.272 |
| 2048 | 900.108 / 898.671 | 899.390 |

Denoise LLM / `action_expert` decode 阶段：

这里 `suffix_len` 才是有效维度，因为 decode 的输入就是 suffix/action expert 序列，mask shape 也会变成 `[1, suffix_len, prefix_len + suffix_len]`。

| prefix_len | suffix_len | mean ms | median ms | min ms | max ms |
|---:|---:|---:|---:|---:|---:|
| 512 | 32 | 12.511 | 12.424 | 12.313 | 12.887 |
| 512 | 64 | 15.000 | 14.887 | 14.791 | 15.474 |
| 1024 | 32 | 19.481 | 19.324 | 19.129 | 20.615 |
| 1024 | 64 | 23.248 | 23.125 | 22.909 | 23.616 |
| 2048 | 32 | 38.381 | 38.409 | 37.857 | 38.751 |
| 2048 | 64 | 49.434 | 49.434 | 48.916 | 50.493 |

HBM 矩阵已编译到本机 cache，板端部署到 `/data/edgefm_smolvla_phase1_benchmark`：

| case | prefill HBM | decode HBM |
|---|---:|---:|
| p512_s32 | 164 MiB | 103 MiB |
| p512_s64 | 164 MiB | 106 MiB |
| p1024_s32 | 203 MiB | 107 MiB |
| p1024_s64 | 203 MiB | 110 MiB |
| p2048_s32 | 333 MiB | 110 MiB |
| p2048_s64 | 333 MiB | 116 MiB |

原始日志和结构化结果保存在本机临时目录：

```text
.tmp_codex/smolvla_phase1_benchmark/results/board_benchmark_all_raw.log
.tmp_codex/smolvla_phase1_benchmark/results/board_benchmark_summary.json
```

## 部署配置提示

如果 runtime 和 export 在同一个用户环境中执行，`engine.tune()` 写入的 backend artifact cache 会记录 `compile_spec.json` 和 stage artifact path。

如果要把 HBM 和配置移动到另一台机器，建议把 `compile_spec["artifact"]` 写入 engine config 的 `_edgefm_internal.backend_artifact`，并确保其中 `metadata.stages[*].artifact_path` 指向目标机器上的 `smolvla_prefill.hbm` 和 `smolvla_decode.hbm`。

示例结构：

```json
{
  "_edgefm_internal": {
    "backend_artifact": {
      "backend": "horizon",
      "artifact_type": "hbm",
      "artifact_path": "/path/to/model.hbm",
      "manifest_path": "/path/to/compile_spec.json",
      "metadata": {
        "stages": [
          {
            "name": "prefill",
            "artifact_path": "/path/to/smolvla_prefill.hbm"
          },
          {
            "name": "decode",
            "artifact_path": "/path/to/smolvla_decode.hbm"
          }
        ]
      }
    }
  }
}
```

实际 metadata 中还会包含 generated module、stage I/O、factory kwargs 等字段。迁移时不要删这些字段，只需要按部署路径修正 artifact path。
