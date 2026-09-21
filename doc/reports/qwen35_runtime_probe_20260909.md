# Qwen3.5 Runtime Probe

## Subsequent Recovery

The 4.57.3-only limitation below is historical. Transformers 5.12.1 and
Hugging Face Hub 1.5.0 were obtained in a separate NAS runtime (`qwen-runtime-002`),
without modifying the shared OpenPI environment. Both checkpoint configurations
and processors loaded. Qwen3.5-0.8B completed image+text generation in
`runs/qwen35-smoke-20260909-002/smoke.json`, PID 4067372, H20 GPU0 UUID
`GPU-ae321531-504c-f416-e693-94fa02779130`, report SHA256
`f7640bc22e30b5f6cc032715de23298e425b76e0b6de694eb28c43c7776a8254`.
This is an official Python smoke using a LeRobot architecture diagram, not a
dataset evaluation or VLAForge deployment. It used the upstream Torch fallback
for linear attention, not the optional accelerated FLA/causal-conv1d path.

The first two formal launch attempts (PIDs 4070833/4070834) exited before
workers started: the launcher pre-created directories rejected by the tool's
create-only contract. Neither constitutes a formal measurement. The old runner
also launched all workers concurrently on one GPU and used unsynchronized TTFT;
it must not be used for the requested independent latency protocol.

2026-09-09. This is an implementation-availability probe on
`zzm-h20-x8-2`; it is not a Qwen3.5 inference or performance result.

## Assets

The previously audited model snapshots are present on the shared NAS:

| Model | Path | Revision | Weight bytes |
|---|---|---|---:|
| Qwen3.5-0.8B | `/xs-train-nas/zzm/models/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | 1,746,942,600 |
| Qwen3.5-2B | `/xs-train-nas/zzm/models/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | 4,548,221,488 |

The asset audit is `doc/reports/qwen35_asset_audit_20260909.md` and its
report SHA256 is
`fed0a8c669974cf0df96903cccd167998eaa0519c99df3e08f5351e788bc53fa`.

## Runtime Probe

The probe used the existing Python 3.11 OpenPI environment:

- Python 3.11.16;
- PyTorch 2.10.0+cu128;
- CUDA 12.8;
- H20 CUDA was available;
- isolated Transformers 4.57.3, Tokenizers 0.22.2 and Safetensors 0.8.0
  installed from the NAS wheelhouse under
  `runs/recovery-20260909/qwen-runtime-001/site`.

Both local model configs declare:

```text
model_type = qwen3_5
architectures = [Qwen3_5ForConditionalGeneration]
```

`transformers.AutoConfig.from_pretrained(..., local_files_only=True)` rejects
both snapshots with:

```text
ValueError: The checkpoint ... has model type `qwen3_5` but Transformers does not recognize this architecture.
```

The isolated wheel itself imports successfully as Transformers 4.57.3, but it
does not provide `Qwen3_5Config`, `Qwen3_5ForConditionalGeneration` or
`Qwen3_5ForCausalLM`. The older installed Transformers 4.53.2 was not used for
model loading either, because it predates the same architecture support.

## Status

No model weights were loaded, no image or text input was executed, and no GPU
benchmark was started during this probe. The correct status is
`pending-implementation`, not `passed`, `unsupported-model`, or
`inference-complete`.

The next valid step requires an exact Qwen3.5 model implementation compatible
with these configs, including the mixed linear-attention/full-attention state,
vision processor, prefill and autoregressive decode. A text-only fallback,
Qwen2.5 implementation, or manual config relabeling would not satisfy the
paper requirement and must not be used.
