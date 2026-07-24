# VLAForge Frontend v0: Real-model Audit

## Scope

This report records the Milestone B evidence for the restricted PyTorch
frontend. It is an exportability and semantic-boundary audit, not a claim that
the exported programs are already packaged as final no-Python C++ artifacts.
The latter is covered by Milestones F and G.

The frontend deliberately captures coarse, pure `TensorRegion` functions. It
does not attempt to compile tokenization, image decoding, Python model classes,
or the VLA control program itself into a tensor graph.

Audit schema: `vlaforge.frontend_model_audit/1`.

## Frontend contract

The implemented frontend:

- consumes declared `TensorRegion` signatures and `@tensor_region` metadata;
- captures regions through `torch.export` without an eager fallback;
- requires bounded profiles for every dynamic tensor dimension;
- audits graph effects for external mutation, hidden RNG, and external I/O;
- permits invocation-local mutable workspaces only when alias analysis proves
  that they do not alias a user input or module value;
- treats evaluation-mode dropout with `train=false` as deterministic;
- rejects mutable plain-function closures;
- produces versioned capture evidence, unsupported reports, backend compile
  requests, and artifact contracts;
- lifts state only from explicit cross-tick source evidence.

## SmolVLA

### Provenance

- Policy: `examples/smolvla/SmolVLA-Base`
- VLM: `examples/smolvla/SmolVLM2-500M-Video-Instruct`
- LeRobot revision: `8fff0fde`
- Policy weight SHA-256:
  `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`
- PyTorch: `2.6.0+cu126`
- Audit device: `cuda:0`

### Region results

| Region | Role | Graph nodes | Export time | Eager/export max abs error | Effect audit |
|---|---|---:|---:|---:|---|
| `prepare_prefix` | image/language/state prefix and KV preparation | 2,965 | 2.107 s | 0 | pass |
| `solver_step` | one bounded flow-matching solver step | 2,525 | 1.569 s | 0 | pass |
| `trim_action_chunk` | action-dimension projection | 5 | 0.028 s | 0 | pass |

The 16-layer prefix KV is flattened into 32 explicit tensor ABI values.
Prefix KV and the solver sample are invocation-local. The solver sample is a
loop-carried SSA value and is not a `StateSlot`. Noise is an explicit region
input, so the captured graph contains no hidden random operation.

The only source-retained cross-tick values are:

- `action_queue`: the generated action chunk reused by later control ticks;
- `queue_cursor`: the next action index within that chunk.

The Semantic IR model now reads, stages, and atomically commits both values on
each tick. The real-model gate compares three consecutive IR queue outputs to
LeRobot's three consecutive `select_action()` results.

### Reproduction

```bash
PYTHONPATH="$PWD/vlaforge/python:/home/zhangzimo/Repos/public/lerobot-v0.4.4/src" \
  /home/zhangzimo/miniconda3/envs/horizon_quant/bin/python \
  vlaforge/tools/audit_real_smolvla_frontend.py \
  --policy-path examples/smolvla/SmolVLA-Base \
  --vlm-path examples/smolvla/SmolVLM2-500M-Video-Instruct \
  --revision 8fff0fde \
  --device cuda:0 \
  --report /tmp/vlaforge-smolvla-frontend.json
```

## OpenVLA

### Provenance

- Checkpoint:
  `/home/zhangzimo/.cache/vlaforge/openvla-7b`
- Revision:
  `47a0ec7fc4ec123775a391911046cf33cf9ed83f`
- Shard SHA-256:
  - `model-00001-of-00003.safetensors`:
    `10d8636256018712c5e5c823d12e22b5797f99bb721bd123bf6bf2379892be85`
  - `model-00002-of-00003.safetensors`:
    `2050b14f21d48904d269f48d5a980fecea87cd7b36641d9b0f015e72d1fe216a`
  - `model-00003-of-00003.safetensors`:
    `ea65305a1577f36f721965bf84c8caec0a948ce7ce84d754701637376c531fef`
- PyTorch: `2.6.0+cu126`
- Audit device/precision: CPU/BF16
- Observed resident memory during the complete audit: approximately 22.3 GB

### Why the audit uses BF16 CPU

The existing real semantic gate uses NF4 on the 12 GiB GPU. PyTorch 2.6
FakeTensor export cannot currently construct the bitsandbytes
`Params4bit` tensor subclass and fails with:

```text
RuntimeError: Creating a new Tensor subclass Params4bit but raw Tensor already
associated to FakeTensor
```

This is a frontend/backend compatibility limitation rather than evidence for
hidden VLA state. The same pinned checkpoint is therefore loaded in BF16 on
CPU for the export audit. NF4 and BF16 numeric results are not claimed to be
identical.

### Region results

| Region | Role | Graph nodes | Export time | Eager/export max abs error | Effect audit |
|---|---|---:|---:|---:|---|
| `generate_action_tokens_prefill` | multimodal prefill and initial KV | 4,427 | 6.586 s | 0 | pass |
| `generate_action_tokens_decode_step` | one cached autoregressive step | 3,378 | 4.510 s | 0 | pass |
| `detokenize_action` | token-to-continuous-action conversion | 17 | 0.093 s | 0 | pass |

The frontend does not export Hugging Face `generate()` as an opaque Python
operation. It separates prefill from a pure decode step and composes the
decode step in an explicit, fixed seven-token loop. The loop produced:

```text
31857, 31864, 31900, 31840, 31860, 31868, 31872
```

The official BF16 `model.generate()` path produced the same seven token IDs.
The 32-layer KV is flattened into 64 explicit loop-carried tensors. It is not
persistent policy state, and the OpenVLA program has no `StateSlot`.

### Reproduction

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHONPATH="$PWD/vlaforge/python" \
  /home/zhangzimo/.venvs/vlaforge-openvla/bin/python \
  vlaforge/tools/audit_real_openvla_frontend.py \
  --checkpoint /home/zhangzimo/.cache/vlaforge/openvla-7b \
  --revision 47a0ec7fc4ec123775a391911046cf33cf9ed83f \
  --unnorm-key bridge_orig \
  --cpu-threads 16 \
  --report /tmp/vlaforge-openvla-frontend.json
```

## Gate B conclusion

Both real checkpoints produce passing, versioned frontend audit reports. The
major model computation is in exported regions, persistent state is completely
visible in Semantic IR, and the effect audit reports no hidden mutation, RNG,
or I/O. Unsupported paths remain explicit and structured.

This gate establishes the source-to-export boundary. It does not establish
AOT artifact backend support or no-Python execution; those remain required
Milestones F and G.
