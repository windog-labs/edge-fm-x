# VLAForge IR: real OpenVLA evidence

> **Historical real-model evidence.** This run predates Invocation IR v0.2;
> it is not a current real compiled-artifact or no-Python Session claim.

Date: 2026-07-23

## Provenance

- Checkpoint: official `openvla/openvla-7b`
- Revision: `47a0ec7fc4ec123775a391911046cf33cf9ed83f`
- Weight shards:
  - `model-00001-of-00003.safetensors`: 6,948,961,960 bytes,
    SHA256 `10d8636256018712c5e5c823d12e22b5797f99bb721bd123bf6bf2379892be85`
  - `model-00002-of-00003.safetensors`: 6,971,232,040 bytes,
    SHA256 `2050b14f21d48904d269f48d5a980fecea87cd7b36641d9b0f015e72d1fe216a`
  - `model-00003-of-00003.safetensors`: 1,162,406,824 bytes,
    SHA256 `ea65305a1577f36f721965bf84c8caec0a948ce7ce84d754701637376c531fef`
- PyTorch: `2.6.0+cu126`
- Transformers: `4.40.1`
- Tokenizers: `0.19.1`
- timm: `0.9.10`
- Accelerate: `0.29.3`
- bitsandbytes: `0.49.2`
- Quantized load: NF4, BF16 compute, double quantization
- Device: NVIDIA GeForce RTX 3060, 12 GiB

All three materialized shards were checked against their Git-LFS object IDs
before model loading.

## Deterministic input fixture

- Image: `rgb_coordinate_grid_v1`, RGB `224 x 224`, SHA256
  `ad060d9279ecb1d3470ed58ae114fa1fb2bd5c610588ebbaaf47b360399a27e1`
- Instruction: `pick up the block`
- Prompt: `In: What action should the robot take to pick up the block?\nOut: `
- Processed inputs:
  - `pixel_values`: `(1, 6, 224, 224)`, BF16
  - `input_ids`: `(1, 19)`, int64
  - `attention_mask`: `(1, 19)`, int64

The trailing space after `Out:` intentionally materializes OpenVLA's expected
empty token `29871`. In the pinned source, `predict_action()` appends that token
when absent but does not extend an already supplied attention mask. Supplying
the token in the input keeps the official method and attention mask unchanged;
no model-source patch is applied.

## Source-derived IR boundary

The real `predict_action` path is represented as:

```text
sample(image, instruction_tokens, instruction_mask)
  -> generate_action_tokens
  -> detokenize_action
  -> validate
  -> commit and publish
```

The Hugging Face autoregressive generation call is a pure TensorRegion. Its KV
cache is invocation-local and is not retained across `predict_action()` calls,
so the IR declares no persistent OpenVLA state.

## Results

| Check | Result |
| --- | ---: |
| Generated action tokens | `31904, 31935, 31852, 31911, 31938, 31865, 31744` |
| Eager-versus-IR token equality | exact |
| Action shape | `(7,)` |
| Maximum eager-versus-IR action error | `0.0` |
| IR trace events | 10 |
| Peak CUDA allocation | 4,509.085 MiB |
| Gate | **passed** |

The eager and IR actions were both:

```text
[-0.0073663630, -0.0208209912, 0.0122303694, -0.0244054615,
 -0.0516903156,  0.0096040771, 0.9960784314]
```

Observed eager and IR timings are retained in the machine-readable local
evidence but are not treated as a performance comparison.

## Reproduction

Use the exact OpenVLA command in `vlaforge/README.md`. The runner writes:

- `artifacts/vlaforge_ir/openvla/real_model_report.json`
- `artifacts/vlaforge_ir/openvla/ir_trace.json`

The formal opt-in pytest gate completed with `1 passed` on this environment.
