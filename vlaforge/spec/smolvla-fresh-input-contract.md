# SmolVLA Fresh-Chunk Input Contract

`adapters/smolvla/smolvla_fresh.py` declares an Adapter-owned tensor profile through the
shared `InvocationBuilder`. Core IR, Plan and C++ execution do not branch on the
model name. This profile is not the legacy action-queue fast path.

## Interface

`build_smolvla_fresh_program(policy, batch, noise)` accepts an already strictly
loaded policy and explicit tensors. The caller owns dataset provenance,
tokenization, state normalization and saved noise. It does not create camera
masks, infer normalization statistics, refit them to a dataset or load weights.

- Every present configured camera has its actual Boolean `[batch]` padding mask.
  Missing configured cameras remain absent; upstream `empty_cameras` semantics
  apply without inventing an additional observed view.
- State and image tensors are floating; language tokens are int64 with Boolean
  masks. Saved noise is FP32 `[batch, chunk_size, max_action_dim]`.
- All inputs are finite, contiguous and on the same device with the same batch
  size. Shape, dtype, device and available camera set form a fixed profile.
- `n_obs_steps=1`, `use_cache=True`, no RTC hidden history and no Pi-Aloha
  transforms are currently required. Unsupported profiles fail explicitly.
- The result contains a complete `[batch, chunk_size, action_dim]` reference,
  input clones, input bindings and example arguments for all four regions.
  The output port declares the actual device.
- Rebinding accepts a new Mapping and saved noise with the same profile.
  Optional revisions use declared port aliases (`image_0`, `image_mask_0`,
  `state`, `instruction_tokens`, `instruction_mask`, `noise`), not raw batch keys.

The prefix cache automatically depends on every declared prefix input,
including every actual camera and mask. It does not depend on the fresh noise.
No cross-invocation action queue is declared. The solver carries both the FP32
sample and an int64 device index. It gathers a precomputed FP32 timestep table
equal to `FP32(1.0 + python_step * (-1.0 / N))`; incremental FP32 accumulation
does not have the same rounding. The update is `sample + (-1.0 / N) * velocity`.
The host induction value is not passed to the solver, so it does not create a
per-step host-to-device timestep upload. The finish region checks the entire
padded sample for nonfinite values before committing the complete action chunk.

Optional fixed vision position specialization verifies every prefix output
against the unspecialized implementation. Failure restores the original vision
embedding module. Capture uses shared `capture_region` and `save_exported_region`
with no eager fallback. Capture success alone is not AOTI or C++ deployment.

## Statistics Gate

`adapters/smolvla_processing.py` resolves explicit MEAN_STD statistics and checks
that an upstream processor consumes them. It does not implement a second
normalization pipeline.

```python
profile = resolve_smolvla_statistics(
    loaded_safetensors,
    namespace=explicit_namespace,
    robot_type=dataset_info["robot_type"],
    feature_shapes={"action": (action_dim,)},
    source_sha256=verified_file_sha256,
)
# Supply profile.stats through the upstream processor's stats override.
profile.require_consumed(postprocessor.steps[0])
native_candidate = postprocessor(reference.clone())
profile.verify_transform("action", reference, native_candidate, inverse=True)
```

Legacy `namespace.buffer.feature.mean/std` keys are mapped only for an explicitly
selected namespace equal to the recorded robot type. Modern unscoped keys use
`namespace=None`. Ambiguous namespaces, missing features or mean/std, wrong
shapes, nonfinite statistics, negative std, unconsumed selected features,
identity mode and ignored statistics all fail closed. Action statistics are
never substituted for state statistics. Numerical scale verification does not
establish robot calibration or physical units.

## Fresh Deployment Tool

`tools/build_real_smolvla_fresh.py` executes `capture`, `compile`, `verify-export`,
`direct` and `session` stages, or all five in that order. Capture requires a v2
saved-observation manifest and explicit `--processing-mode`. It verifies the
checkpoint, every declared processor file and the full recovered-profile
provenance against the selected policy; equal model weights alone do not permit
mixing preprocessing profiles. Strict input records must include successful
actual state-transform checks. Failed numerical gates do not discard raw output.

Four saved/reloaded exports are compared against the complete official reference.
The direct AOTI and generated C++ Session stages retain full output, per-dimension
metrics, exact-byte checks and separate technical and paper numerical gates.
The C++ harness derives all input shapes/dtypes from the shared IR, changes every
observation revision for each fresh invocation and exports the full action chunk.
It runs without a linked Python runtime and with Python environment paths
disabled. `--session-repetitions` checks repeated complete invocation behavior.

Use `--session-label` for a new runtime build or diagnostic attempt; each label
has a separate bundle and Session output directory. Completed or partially
materialized Session outputs cannot be silently overwritten. Recorded cold
resident-tensor times exclude input/output transfers and are not steady-state
or board benchmark claims. Static `*.operators.json` inventories are not runtime
profiles. `*.runtime-inputs.pt` preserve actual exported-region CPU arguments.

Action scale candidates are stored separately from normalized model outputs.
Normalized tensor thresholds do not automatically apply to rescaled units.
Even exact inverse statistics formulas do not establish physical calibration.

## Processing Evidence

The official `lerobot/smolvla_base` files at `c83c3163b8ca9b7e67c509fffd9121e66cb96205`
and their migration commit `4d2f2b37fa245361ef1efe6d91ce96b8bd4af511` contain only
robot-scoped six-dimensional action statistics in both processor state files.
They do not contain `observation.state` statistics despite declaring state
MEAN_STD. The current upstream generic processor silently returns identity for
that missing feature. These five downloaded JSON/statistics files were verified
against official Git/LFS digests and are byte-equal to the local files.

The official pre-migration revision `3326b100334ffc0a0bd1ec27e3afb1cfa2a6000c`
also declares STATE=MEAN_STD, and its bounded safetensors header contains
`normalize_inputs.<robot>_buffer_observation_state.mean/std` for all three robot
namespaces. This supports a migration-loss diagnosis, not an intentional state
identity contract. The initial bounded-header audit did not verify the full
historical checkpoint. A later independent recovery audit downloaded and
SHA-verified all 906720008 bytes, checked all model tensors bitwise against the
current checkpoint with only the declared compilation-prefix key rename, and
checked every published robot namespace's action statistics. It wrote a separate
canonical SO100 profile without modifying the original published policy. See
`adapters/smolvla_migration.py` and the recovery manifest for this stronger gate.

The first real-input fresh-chunk run is therefore an **as-published processor
compatibility audit**, not verified normalized-observation or calibrated-action
inference. IR, Plan and saved/reloaded exported regions matched its official
reference exactly for one complete 300-element action chunk. An explicit `so100`
action-scale candidate has been consumed by the upstream postprocessor and
verified numerically. Subsequent strict-statistics captures use the explicitly
recovered profile and independently formula-verified actual state transforms.
They do not retroactively upgrade compatibility inputs. Current per-attempt
capture/AOTI/C++ status and numerical gate failures are recorded separately;
fresh compilation is not equivalent to paper numerical or board acceptance.

Reproducible artifacts and the corrected scope are under
`artifacts/edgefm-vla-goal/20260906-044509/fresh-smolvla/README.md`.
