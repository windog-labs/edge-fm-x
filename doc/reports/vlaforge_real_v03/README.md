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

## SmolVLA real L4

The same real prefix/solver/trim packages were combined with five captured
Adapter support Regions (`make_timestep` and four queue operations) in one
verified eight-Region Compile Bundle. The generated no-Python C++ Session ran
on the same RTX 3060 with:

- bit-exact direct-AOTI versus generated-C++ parity for the complete
  `[1, 50, 6]` action chunk;
- one same-revision exact-cache hit and explicit new-revision,
  missing-revision, and episode-reset misses;
- 152 successful primary Session invocations, 304 authoritative state commits,
  and monotonically allocated per-state versions;
- typed C++ and generic C ABI output equality;
- a NaN output-validation failure that aborted the transaction, exposed no
  uncommitted output, and did not increment state versions;
- execution under invalid `PYTHONHOME/PYTHONPATH`, with no `libpython` in
  `ldd`.

The verified static plan assigns 2,314,353 bytes to recomputable derived prefix
cache and 2,440 bytes to authoritative queue/cursor state. These categories
remain semantically separate. Hashes, input fixture identities, bundle digest,
trace counts, and the exact reproduction command are recorded in
`smolvla_artifact_l4.json`.

This upgrades the fixed `SmolVLA-Base` checkpoint to real Host-CUDA L4. It is
not an Orin claim and the recorded audit runtime is not yet a paper-grade
latency benchmark.
