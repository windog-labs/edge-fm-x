# VLAForge IR v0.1 coverage and abstraction gaps

> **Archived v0.1 report.** Current model coverage and evidence levels are in
> `doc/model_cards/README.md`.

Date: 2026-07-23

This report audits the deliberately small VLA business IR. It does not assess
a general tensor compiler, workflow engine, or C++ runtime.

## Evidence levels

| Model path | Evidence | Current result |
| --- | --- | --- |
| SmolVLA | real checkpoint, eager-versus-IR solver/action trace | passed |
| OpenVLA | official checkpoint, eager-versus-IR token/action trace | passed |
| π0 / π0.5 | pinned LeRobot source audit | mapping passed; no checkpoint claim |
| flow/autoregressive fixtures | deterministic offline tests | passed; not real-model evidence |

## Business-semantic coverage

| VLA concern | Core representation | SmolVLA | OpenVLA | π0 / π0.5 |
| --- | --- | ---: | ---: | ---: |
| timestamped image/language/proprioception | `vla.sample_input` + epoch | yes | yes | yes |
| source-retained action queue | `StateSlot` + snapshot/staged write | yes | not present | yes |
| queue refill versus reuse | `vla.if` | yes | not needed | yes |
| prefix/context computation | pure TensorRegion result | yes | yes | yes |
| flow/diffusion solver value | `vla.for` carried SSA | yes | not present | yes |
| autoregressive action tokens | pure bounded generation region or `vla.for` | not present | yes | not present |
| detokenization/action decoding | pure TensorRegion | yes | yes | yes |
| action validity and visibility | validate + transaction commit + publish | yes | yes | yes |
| episode reset | reset policy / `vla.reset` | yes | no state | yes |

All three paths compose the same generic opcodes. No core opcode contains a
model name.

## Persistent-state inventory

The admission rule is source behavior, not compiler convenience.

| Value | Persistent? | Reason |
| --- | ---: | --- |
| action queue and cursor | yes when the policy retains them | affects later control ticks |
| explicit RNG state | only when retained or supplied by contract | required for deterministic replay |
| observation/history cache | only when retained by the source path | may affect later policy calls |
| prefix KV inside one chunk/generate call | no | invocation-local implementation state |
| flow solver sample `x_t` | no | bounded loop-carried SSA |
| OpenVLA transformer KV inside `generate()` | no | not retained across `predict_action()` calls |
| invented previous action/history | no | no source provenance |

## Deliberately unsupported in v0.1

- arbitrary Python control flow or arbitrary dynamic task graphs;
- tokenizer, PIL, camera I/O, network I/O, and robot transport inside Tensor
  Regions;
- a new tensor/operator dialect;
- distributed futures, a general workflow scheduler, or a hard real-time
  language;
- model-named operations or quantization-specific VLA operations;
- C++ AOT runtime and backend kernel code.

`vla.while`, `vla.async`, and `vla.await` retain minimal goal-required
reference semantics, but are compatibility-only and are not active paper
claims.

## Remaining abstraction gaps

1. **Frontend automation.** Real adapters currently establish TensorRegion
   boundaries explicitly. Automatic `torch.export`, hidden-state audit, and
   source-to-IR lifting remain future work.
2. **OpenVLA token-step visibility.** The validated official Hugging Face
   `generate()` path is one pure region. This proves the VLA-level contract,
   but does not yet expose per-token scheduling or KV memory planning.
3. **Shape and structured-value coverage.** v0.1 supports the shapes and
   scalar/tensor values needed by the validation models; broader symbolic
   shape algebra and nested robot observations are not implemented.
4. **Scheduled lowering.** Logical state-slot capacity is analyzed, but no
   generated C++ arena or backend execution plan is part of this foundation
   milestone.
5. **Held-out execution.** π0/π0.5 validates abstraction reuse at source level
   only until a real checkpoint is run.

None of these gaps requires expanding the public VLA IR before there is a real
model path and measurable deployment need.

## Freeze decision

The v0.1 core is sufficient for the two required model structures and the
held-out π0/π0.5 source mapping. New public operations are frozen behind the
source-location, composition, verifier, and test admission rule in
`vlaforge/spec/vla-profile.md`.
