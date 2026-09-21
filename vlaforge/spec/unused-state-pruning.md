# Unused Exported State

`analysis.unused_state.prune_unused_state` is an offline, model-independent
storage pass on a static, pure `torch.export.ExportedProgram`. It removes unused
lifted parameter/persistent-buffer placeholders and state-table entries not named
by any signature input. It does not remove computation, user inputs, output
ports, or change a scheduler, timestep, dtype, layout or arithmetic operation.
Nested graph computations remain unchanged. New models use the same pass.

The pass requires a successful recursive effect audit. Dynamic profiles,
unlifted module calls/attributes, custom constants and module-call signatures
that reference removable state are rejected. Tensor constants remain bound.
Aliased storage is counted once; removing an unused alias name does not establish
that its underlying allocation is freed.

## API And Ownership

```python
result = prune_unused_state(ep, source_artifact_sha256=verified_archive_sha256)
ledger = save_pruned_state(result, new_output_directory)
```

The caller verifies the archive digest before loading and keeps source/retained
tensors immutable until serialization completes. The result shares retained
tensor storage with the source; it is not a deep copy of all weights. Release
the original EP separately when measuring process memory. The supplied source
digest is a provenance assertion, not proof that an arbitrary in-memory EP was
loaded from that archive.

Saving never overwrites an existing directory. Before and after serialization it
checks the complete retained tensor/constant hashes, layouts and unique-storage
inventory, nested graph text, graph input/output signature, module-call and
pytree structure, plus an independently sealed ledger digest. Version 2 adds
the latter signature/ledger guards. Version 1 evidence remains version 1; it
cannot inherit checks added after its frozen source was executed.

## Evidence Boundary

The ledger records removed and retained state, source/result storage groups,
source and saved archive digests. It deliberately sets
`runtime_peak_reduction_verified=false` and `full_model_output_verified=false`.
Logical/storage-table byte counts are not allocator peak measurements, and
serializing an EP is not compilation or a no-Python deployment result.

`tools/prune_exported_state.py` additionally verifies each complete serialized
Region example against the original, in-memory pruned and saved/reloaded pruned
EP. It binds actual numerical context v2, source/tool/pass digests and full raw
output archives. The caller must explicitly acknowledge process-global numerical
restoration. It preserves saved device placement and does not remap CPU scalar
constants to CUDA. The tool's GPU ownership handshake precedes Torch import and
any model loading.

Full-model acceptance requires a separate fresh-process experiment using all
saved Region artifacts, original inputs/noise and scheduler, and complete final
outputs. Peak-memory claims require paired allocator measurements over equivalent
phases. Compiled/C++ artifacts must be rebuilt and revalidated independently.

## Current Verification

The H20 experiment `openpi-inventory/h20-unused-state-001` uses frozen version 1.
All four actual OpenPI Region examples pass byte equality, including all 37
prefix outputs. Prefix state/constants storage changes from 7,011,418,256 to
5,848,772,624 bytes; step storage changes from 7,011,417,648 to 636,001,104 bytes.
Time/finite have no state. An independent local audit checks all twelve saved
output trees, 132 frozen source files and sampled GPU ownership. It does not
reload the large pruned weight archives locally or measure full-model peaks.

Current version 2 CPU regression has 21 passing tests including the GPU monitor
tests. Version 1 actual H20 output evidence is not relabeled as version 2.
