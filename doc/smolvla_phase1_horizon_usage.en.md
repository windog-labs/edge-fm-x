# SmolVLA Phase-1 Horizon Usage Guide

This document explains how EdgeFM currently supports the LLM portion of SmolVLA, namely:

- `prefill`: takes prefix embedding/mask/position as input, and outputs the per-layer prefix KV cache.
- `decode`: takes suffix embedding/mask/position as input, uses the prefix KV cache, and outputs `expert_hidden`.

The current scope does not include ViT, `embed_suffix`, or `action_out_proj`. These modules should prepare inputs or consume outputs outside of EdgeFM.

Important prerequisites:

- The `smolvla` referred to here is **LeRobot's `SmolVLAPolicy`**, which contains `vlm_with_expert`, that is:
  - VLM prefix 16 layers
  - Action Expert 16 layers
- It is **not** a matter of crudely splitting a standalone `SmolVLM2-500M-Video-Instruct` text/multimodal backbone into the two stages `prefill` and `decode`.
- A standalone SmolVLM2 checkpoint such as `SmolVLM2-500M-Video-Instruct` only has the VLM backbone, without SmolVLA's `action_expert`, `embed_suffix`, and `action_out_proj`, so it cannot directly serve as the complete source model for the phase-1 two-stage export described in this document.

## Confirmed LeRobot Version and Model

You should currently use the LeRobot policy checkpoint as the entry point:

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

Note:

- The `model.safetensors` of `lerobot/smolvla_base` contains the `SmolVLAPolicy` weights, which is why it is the checkpoint needed for the phase-1 export.
- `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` is merely the name of the VLM backbone in that policy config, and cannot replace `lerobot/smolvla_base` on its own.
- The current `horizon_quant` uses Python 3.10. LeRobot main / v0.5.x already requires Python 3.12, so it cannot directly serve as the Horizon compilation source entry point. It is recommended to use a LeRobot `v0.4.4` checkout, which is still Python 3.10 compatible and includes `SmolVLAPolicy/vlm_with_expert`.

Locally, it is recommended to prepare:

```bash
git -C ~/Repos/public/lerobot worktree add ~/Repos/public/lerobot-v0.4.4 v0.4.4

source ~/miniconda3/bin/activate horizon_quant
pip install 'draccus==0.10.0' 'einops>=0.8.0,<0.9.0' \
  'gymnasium>=1.1.1,<2.0.0' 'diffusers>=0.27.2,<0.36.0'
```

If a direct connection to Hugging Face times out, you can use a mirror to download the policy checkpoint:

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

`examples/smolvla/SmolVLA-Base` is already covered by the repository's `.gitignore`, and model weights should not be committed.

If you already have the standalone VLM backbone locally, for example `examples/smolvla/SmolVLM2-500M-Video-Instruct`, you can avoid downloading the backbone again from the Hub during export:

```bash
export EDGE_FM_SMOLVLA_VLM_MODEL_PATH=$PWD/examples/smolvla/SmolVLM2-500M-Video-Instruct
export EDGE_FM_SMOLVLA_DEVICE=cpu
export EDGE_FM_LEROBOT_ROOT=/home/zhangzimo/Repos/public/lerobot-v0.4.4
```

`EDGE_FM_SMOLVLA_VLM_MODEL_PATH` only overrides `vlm_model_name` in the policy config; it does not change the loading of the policy/action expert weights of `lerobot/smolvla_base`.

## Interface Principles

SmolVLA does not use a model-specific `smolvla_prefill` or `smolvla_decode` public API. It reuses EdgeFM's generic tensor stage interface:

```cpp
TensorMap EdgeFM::prefill(int32_t request_id, const TensorRefMap& inputs) const;
TensorMap EdgeFM::decode(int32_t request_id, const TensorRefMap& inputs) const;
```

The corresponding Python bindings are:

```python
prefill_outputs = engine.prefill(request_id, inputs)
decode_outputs = engine.decode(request_id, inputs)
```

`generate()` is still the token generation API, primarily serving Qwen LLM/VLM sampling-style generation. The SmolVLA phase-1 denoise expert is a tensor-in/tensor-out forward, so it goes through the `decode()` stage rather than `generate()`.

## Exporting the Two Horizon Models

SmolVLA phase-1 needs to export two HBMs:

- `smolvla_prefill.hbm`: responsible for prefix prefill, outputting `prefix_kv_layer_*`.
- `smolvla_decode.hbm`: responsible for suffix/action expert decode, taking `suffix_*` and `prefix_kv_layer_*` as input, and outputting `expert_hidden`.

Step one, prepare the engine config. Example of key fields:

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

Step two, generate the Horizon compile spec:

```bash
python - <<'PY'
import edge_fm

engine = edge_fm.EdgeFM("smolvla_horizon_engine.json")
engine.tune()
PY
```

`tune()` writes out `compile_spec.json` and prints in the log:

```text
Horizon compile spec written: /path/to/compile_spec.json
```

If you need to manually find the most recent spec, you can also use:

```bash
find ~/.cache/edge-fm/backend_artifacts -name compile_spec.json -printf '%T@ %p\n' \
  | sort -n \
  | tail -1
```

Step three, compile the `prefill` and `decode` stages separately. The EdgeFM helper can do this directly:

- generated module initialization
- ONNX export
- J6M rewrite diagnostic file generation
- `hb_compile` YAML generation
- `hb_compile` invocation
- copying the HBM to the `artifact_path` of the corresponding stage in `compile_spec.json`

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

If you have already exported the ONNX and do not want to re-trace, you can reuse the existing ONNX:

```bash
python scripts/horizon/compile_horizon_from_spec.py "$SPEC" \
  --stage prefill \
  --horizon-rewrite on \
  --skip-model-init \
  --hb-compile \
  --reuse-onnx \
  --onnx-path "$PWD/.tmp_codex/smolvla_phase1/smolvla_prefill.onnx"
```

When `--compiler-command` is not passed or `--dry-run` is added, the helper will not invoke the external compiler and will only generate the preparation manifest, which is convenient for inspecting the stage I/O:

```bash
python scripts/horizon/compile_horizon_from_spec.py "$SPEC" --stage prefill --dry-run
python scripts/horizon/compile_horizon_from_spec.py "$SPEC" --stage decode --dry-run
```

`--stage` only accepts `prefill` and `decode`. The old `expert_denoise` name is no longer used as a public stage name.

Default Horizon compilation parameters:

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

The SmolVLA generated adapter uses `EDGE_FM_SMOLVLA_EXPORT_DTYPE=float32` by default, which converts the LeRobot policy to FP32 before export. This default is necessary: when the native BF16 is kept, the decode stage triggers a `FLOAT vs BFLOAT16` type conflict during HMCT shape inference. If you only do PyTorch diagnostics, you can set `EDGE_FM_SMOLVLA_EXPORT_DTYPE=keep` to keep the model's original dtype.

## Stage I/O

The actual shape and dtype are governed by the `stages` in `compile_spec.json`. The default generation rules are as follows.

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

The current Horizon tensor stage runtime expects the input `Tensor` to be on the CPU, with its shape/dtype exactly matching the HBM runtime I/O.

The mask inputs use `uint8` 0/1 representation, and the generated adapter converts them to `torch.bool` before calling LeRobot's `vlm_with_expert.forward()`. This preserves the bool mask semantics of LeRobot's `make_att_2d_masks()` while avoiding the introduction of an extra bool dtype into the EdgeFM public Tensor API.

The public I/O dtype of the KV cache and `expert_hidden` is currently fixed to `float32`. LeRobot internally uses the backbone weight dtype (currently bfloat16), and the generated adapter performs conversions at the stage boundaries: prefill output KV is converted to `float32`, decode input KV is converted back to the model weight dtype, and the decode output `expert_hidden` is converted to `float32`. This sidesteps the current limitation that the Horizon runtime public Tensor path does not support BF16.

## Python Invocation Example

The example below assumes that `embed_suffix` has already been completed outside of EdgeFM, and that the input numpy arrays' shape/dtype are consistent with `compile_spec.json`.

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

In the example, `copy_data=True`, so the returned `Tensor` owns its own CPU buffer. If switched to a zero-copy view, the caller must ensure the numpy buffer stays alive until after `prefill()` or `decode()` returns.

Under the same `engine` and the same `request_id`, `prefill()` caches `prefix_kv_layer_*` inside EdgeFM, so `decode()` can pass only the suffix-related inputs.

If you need to explicitly pass the KV cache, for example across processes, across engines, or when you want to hand the KV over to external lifecycle management, then also add the `prefix_kv_layer_*` from `prefill_outputs` to the decode inputs:

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

## C++ Invocation Example

The example below only shows the structure of the EdgeFM API. In real business code, the data pointer, shape, and dtype of `Tensor::view` or `Tensor::clone_from` should come from your input buffers.

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

If you do not rely on the internal request cache, you can also explicitly pass the prefill output KV as decode input:

```cpp
for (auto& item : prefill_outputs) {
    if (item.first.rfind("prefix_kv_layer_", 0) == 0) {
        decode_inputs[item.first] = &item.second;
    }
}
```

Note: when explicitly passing KV, `prefill_outputs` must stay alive until the `engine.decode()` call finishes, because `TensorRefMap` only holds `Tensor*`.

## J6M Build and Verification Status

The following flow was verified on 2026-04-27.

Host HBM compilation environment:

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

Measured artifacts:

```text
compile_spec: /home/zhangzimo/.cache/edge-fm/backend_artifacts/c86aba310f8c259c/compile_spec.json
prefill HBM: /home/zhangzimo/.cache/edge-fm/backend_artifacts/c86aba310f8c259c/smolvla_prefill.hbm
decode HBM: /home/zhangzimo/.cache/edge-fm/backend_artifacts/c86aba310f8c259c/smolvla_decode.hbm
prefill ONNX: .tmp_codex/smolvla_phase1/smolvla_prefill.onnx
decode ONNX: .tmp_codex/smolvla_phase1/smolvla_decode.onnx
prefill hb_compile log: .tmp_codex/smolvla_phase1/prefill_hb_compile.log
decode hb_compile log: .tmp_codex/smolvla_phase1/decode_hb_compile.log
```

J6M Docker cross-build:

```bash
EDGE_FM_BUILD_JOBS=4 bash scripts/docker/build_hrz.sh install
```

`scripts/docker/build_hrz.sh` automatically mounts when the OpenExplorer v3.5.0 deps exist on the local machine:

```text
host: ~/Packages/horizon_j6_open_explorer_v3.5.0-py310_20250927/samples/ucp_tutorial/deps_aarch64
container: /opt/horizon_deps_aarch64
```

and enables `ENABLE_HORIZON_RUNTIME=ON` by default. If the deps are at another path, use:

```bash
EDGE_FM_HOST_HORIZON_DEPS_ROOT=/path/to/deps_aarch64 \
EDGE_FM_BUILD_JOBS=4 \
bash scripts/docker/build_hrz.sh install
```

After the build completes, `build-j6m/install/lib/libedge_fm.so` is an aarch64 artifact and links directly against the Horizon runtime:

```text
libdnn.so
libhbucp.so
libhbrt4.so
libhbtl.so
libhb_arm_rpc.so
...
```

On-board `j6m-1` verification:

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

dummy raw tensor perf smoke:

```text
prefill: frame latency 17.649 ms, 56.449 FPS, frame_count=1, thread_num=1
decode: frame latency 12.125 ms, 81.867 FPS, frame_count=1, thread_num=1
```

The EdgeFM C++ API smoke has also been verified on `j6m-1`. This smoke does not go through `hrt_model_exec`, but instead links directly against `libedge_fm.so` and calls `EdgeFM::prefill()` and `EdgeFM::decode()`:

```bash
ssh j6m-1 "cd /userdata/edgefm_smolvla_phase1 && \
  export EDGE_FM_CONFIG_DIR=/userdata/edgefm_smolvla_phase1/config && \
  export LD_LIBRARY_PATH=/userdata/edgefm_smolvla_phase1:/data/apps/hrt_model_exec/aarch64/lib:\$LD_LIBRARY_PATH && \
  ./edge_fm_smolvla_horizon_smoke \
  /userdata/edgefm_smolvla_phase1/smolvla_horizon_engine_board.json"
```

Output confirmation:

```text
SmolVLA Horizon prefill/decode smoke passed
prefill outputs: 16
expert_hidden shape: [1, 50, 720]
```

This smoke verifies two things about the edge-fm runtime:

- `prefill()` can load `smolvla_prefill.hbm` and return 16 `prefix_kv_layer_*`.
- `decode()` can reuse the KV cached by prefill under the same `request_id`, and invoke `smolvla_decode.hbm` to output `expert_hidden`.

This only proves that the HBM can actually execute on the J6M BPU runtime; it does not represent end-to-end policy accuracy. Accuracy verification still requires integrating the real `embed_suffix`, real prefix embeddings/mask/position, the real denoise mask, and `action_out_proj`.

## J6M EdgeFM Performance Data

The following performance data was measured on 2026-04-27 via the EdgeFM C++ API on the `j6m-1` board, without going through `hrt_model_exec`. The test binary and run location are:

```text
host binary: build-j6m/bin/edge_fm_smolvla_horizon_benchmark
board dir: /data/edgefm_smolvla_phase1_benchmark
board host: j6m-1 (hostname: hobot)
runtime libs: /data/apps/hrt_model_exec/aarch64/lib
warmup: 3
iterations: 20
inputs: dummy CPU tensors, shape/dtype exactly matching the HBM I/O
```

Run command template:

```bash
ssh j6m-1 "cd /data/edgefm_smolvla_phase1_benchmark && \
  export EDGE_FM_CONFIG_DIR=/data/edgefm_smolvla_phase1_benchmark/config && \
  export LD_LIBRARY_PATH=/data/edgefm_smolvla_phase1_benchmark:/data/apps/hrt_model_exec/aarch64/lib:\$LD_LIBRARY_PATH && \
  ./edge_fm_smolvla_horizon_benchmark \
    /data/edgefm_smolvla_phase1_benchmark/p512_s32/smolvla_horizon_engine_board.json \
    --prefix-len=512 --suffix-len=32 --warmup=3 --iterations=20 --stage=both"
```

Timing scope:

- `prefill`: corresponds to SmolVLA's prefix prefill stage, that is, `EdgeFM::prefill()`. The inputs are `prefix_embeds / prefix_attention_mask / prefix_position_ids`, and the output is the 16-layer `prefix_kv_layer_*`. The timing includes the CPU input copy, BPU forward, output KV copy, and the EdgeFM-internal KV cache update.
- `decode`: corresponds to the LLM / `action_expert` stage in SmolVLA's denoise, that is, `EdgeFM::decode()`. Before testing, an untimed `prefill()` is first run to fill the request cache; the decode timing itself includes the suffix input copy, the cached KV injection copy, the BPU forward, and the `expert_hidden` output copy.
- The current test measures phase-1 stage runtime performance, and does not include ViT, `embed_suffix`, the full denoise loop scheduling, `action_out_proj`, or real business pre-processing/post-processing.

Prefill stage:

Here you should only look at `prefix_len`. Each `prefix_len` actually ran two paired cases, namely `suffix_len=32` and `suffix_len=64`. Since prefill itself does not consume the suffix, the main table below gives the representative value of the two cases, taken as the average of the two `mean_ms` values; the original results of each case are still retained in `board_benchmark_summary.json`.

| prefix_len | paired case means ms (`s32 / s64`) | representative mean ms |
|---:|---:|---:|
| 512 | 74.252 / 74.275 | 74.264 |
| 1024 | 248.250 / 248.293 | 248.272 |
| 2048 | 900.108 / 898.671 | 899.390 |

Denoise LLM / `action_expert` decode stage:

Here `suffix_len` is the meaningful dimension, because the input of decode is the suffix/action expert sequence, and the mask shape also becomes `[1, suffix_len, prefix_len + suffix_len]`.

| prefix_len | suffix_len | mean ms | median ms | min ms | max ms |
|---:|---:|---:|---:|---:|---:|
| 512 | 32 | 12.511 | 12.424 | 12.313 | 12.887 |
| 512 | 64 | 15.000 | 14.887 | 14.791 | 15.474 |
| 1024 | 32 | 19.481 | 19.324 | 19.129 | 20.615 |
| 1024 | 64 | 23.248 | 23.125 | 22.909 | 23.616 |
| 2048 | 32 | 38.381 | 38.409 | 37.857 | 38.751 |
| 2048 | 64 | 49.434 | 49.434 | 48.916 | 50.493 |

The HBM matrix has been compiled into the local cache, and deployed to the board at `/data/edgefm_smolvla_phase1_benchmark`:

| case | prefill HBM | decode HBM |
|---|---:|---:|
| p512_s32 | 164 MiB | 103 MiB |
| p512_s64 | 164 MiB | 106 MiB |
| p1024_s32 | 203 MiB | 107 MiB |
| p1024_s64 | 203 MiB | 110 MiB |
| p2048_s32 | 333 MiB | 110 MiB |
| p2048_s64 | 333 MiB | 116 MiB |

The raw logs and structured results are saved in the local temporary directory:

```text
.tmp_codex/smolvla_phase1_benchmark/results/board_benchmark_all_raw.log
.tmp_codex/smolvla_phase1_benchmark/results/board_benchmark_summary.json
```

## Deployment Configuration Tips

If the runtime and export are executed in the same user environment, the backend artifact cache written by `engine.tune()` records `compile_spec.json` and the stage artifact paths.

If you want to move the HBM and config to another machine, it is recommended to write `compile_spec["artifact"]` into the engine config's `_edgefm_internal.backend_artifact`, and ensure that `metadata.stages[*].artifact_path` within it points to `smolvla_prefill.hbm` and `smolvla_decode.hbm` on the target machine.

Example structure:

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

The actual metadata will also include fields such as the generated module, stage I/O, and factory kwargs. When migrating, do not delete these fields; you only need to correct the artifact paths according to the deployment paths.
