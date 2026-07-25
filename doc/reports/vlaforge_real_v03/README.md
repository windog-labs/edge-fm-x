# VLAForge Host CUDA real-artifact evidence

Date: 2026-07-25

This directory records real-checkpoint compiled-artifact evidence. It is
separate from `vlaforge_real_v02`, which only established eager-to-Invocation
IR L2.

## SmolVLA real L3

The real `SmolVLA-Base` checkpoint was captured as three flat TensorRegions:

- `prepare_prefix`: 4 CUDA inputs, 33 outputs (pad mask plus 16 flattened
  BF16 KV key/value pairs), 2,935 exported graph nodes;
- `solver_step`: pad mask, loop-carried sample, timestep, and 32 KV tensors,
  2,545 exported graph nodes;
- `trim_action_chunk`: one sample input and one action-chunk output.

All three exported programs were compiled to `sm_86` AOTInductor packages and
executed on the RTX 3060. Upstream eager and the exported 10-step pipeline were
bit-exact for the final `[1, 50, 6]` action chunk. The packaged artifacts are
deterministic but not bit-exact to eager:

| Measurement | Result |
|---|---:|
| maximum prefix-output NRMSE | 0.01860094 |
| 10th solver-step NRMSE | 0.01323196 |
| final action maximum absolute error | 0.02784944 |
| final action mean absolute error | 0.00802071 |
| final action NRMSE | 0.01303127 |
| repeated artifact execution | bit-exact |

The acceptance contract was fixed at prefix/solver NRMSE `<= 0.02`, final
maximum absolute error `<= 0.05`, and final mean absolute error `<= 0.01`.
Therefore this is real-model L3 numerical parity, not exact parity.

The recorded single-run timings and 1,796.59 MiB peak CUDA allocation are
pipeline audit metadata, not paper-grade benchmark results. Artifact hashes,
sizes, compile times, per-step errors, versions, and tolerances are in
`smolvla_artifact_l3.json`.

Real SmolVLA L4 remains pending until the split artifacts execute through the
generated no-Python C++ Session with device-resident exact prefix cache,
bounded solver SSA, CUDA authoritative state, transactional queue updates, and
failure rollback.
