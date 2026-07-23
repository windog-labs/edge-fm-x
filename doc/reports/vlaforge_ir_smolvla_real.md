# VLAForge IR: real SmolVLA evidence

Date: 2026-07-23

## Provenance

- Policy: `examples/smolvla/SmolVLA-Base/model.safetensors`
- Policy SHA256:
  `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`
- VLM: `examples/smolvla/SmolVLM2-500M-Video-Instruct`
- LeRobot source revision: `8fff0fde`
- PyTorch: `2.6.0+cu126`
- Transformers: `4.57.3`
- Device: NVIDIA GeForce RTX 3060

## Source-derived IR boundary

The real `predict_action_chunk` path is represented as:

```text
sample(batch, explicit_noise)
  -> prepare_prefix
  -> for 10 steps carry solver_sample
  -> trim_action_chunk
  -> validate
  -> commit and publish
```

Prefix KV and the solver sample are local SSA values because LeRobot does not
retain them across policy invocations. The separate `select_action` check
validates the one real persistent behavior: an action chunk is generated when
the policy queue is empty and subsequent control ticks pop queued actions.

## Results

| Check | Result |
| --- | ---: |
| Action shape | `(1, 50, 6)` |
| Solver steps | 10 |
| Maximum eager-versus-IR solver error | `0.0` |
| Maximum eager-versus-IR final action error | `0.0` |
| Queue action errors for indices 0, 1, 2 | `0.0`, `0.0`, `0.0` |
| IR trace events | 19 |
| Peak CUDA allocation | 918.224 MiB |
| Gate | **passed** |

The observed eager and IR timings are retained in the machine-readable local
evidence, but are not reported as a performance comparison: they are single
correctness runs after model loading and do not use a controlled benchmark
protocol.

## Reproduction

Run `vlaforge/tools/run_real_smolvla.py` with the policy and VLM paths above.
The tool writes schema-versioned JSON evidence and a normalized IR trace.
