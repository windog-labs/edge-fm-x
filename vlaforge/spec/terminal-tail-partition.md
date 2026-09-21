# Terminal Tail Partition

`vlaforge.analysis.terminal_tail.partition_terminal_tail` is a static,
source-preserving partition for an explicitly selected terminal tensor suffix.
It does not choose an optimization, rewrite an Invocation, or certify a backend.

```python
result = partition_terminal_tail(
    exported_program,
    inputs=("sample", "velocity", "index"),
    outputs=("next_sample", "next_index"),
    source_artifact_sha256=verified_archive_sha256,
)
```

The result contains an unchanged control EP, a prefix EP, the existing
`TensorSlice` tail, computed frontier names and typed `TailInputRoute` records.
Each route identifies a prefix output index or original flattened USER_INPUT
index. USER_INPUT indices include specialized literal slots, not only tensors.
The prefix keeps the original input pytree and complete input signature, even
when an input is no longer read by its computation. Its outputs are a tuple of
computed frontier tensors. The tail outputs are the original flattened tensor
outputs in their original order; the Adapter retains the external output tree.

## Scope

- The selected closure must be the complete continuous terminal suffix and
  produce every original Tensor USER_OUTPUT. A later assertion or intervening
  unrelated operation prevents partitioning.
- `extract_tensor_slice` supplies closure, static tensor profiles and mutable
  alias checks. Unknown calls, HOPs, unlifted state and implicit RNG are rejected.
- Every nonselected node remains in original order, including unused arithmetic,
  no-output assertions and independent invocation-local workspace writes. No
  dead-code elimination or new tensor math is performed. Constructor-induced
  changes to the retained node records are rejected.
- Computed frontiers must have contiguous zero-offset storage and must not alias
  user inputs, state or one another. Tail outputs must not alias its boundaries
  or one another. `_unsafe_view`, whose schema omits its storage alias, is outside
  this initial contract. Direct lifted-state boundaries and zero-output prefixes
  are unsupported.
- State/constant tensor objects are shared with the source, not duplicated or
  pruned. Logical bytes and tensor metadata are hashed. Node metadata retains
  the original FakeTensor domain. Root output pytree metadata is updated; nested
  module-call signatures referring to removed nodes are rejected.
- Original guard code, complete input signature and retained computation are
  checked. The original source archive digest is caller-verified, not asserted
  as verified merely because a digest string was supplied.

The caller must keep state immutable until serialization. `result.validate()`
checks graph, state, route, signature and ledger identities. The save helper
writes fresh control/prefix EP archives with SHA256/size records after validating
the result, then validates it again. It does not emit a tail artifact: use the
existing capture/export interface with independently recorded boundary inputs
and `result.tail.validate_inputs`. Failed partial directories are not reused.

## Adapter Wiring

Use two ordinary declared TensorRegions and existing `InvocationBuilder.call`
operations. Pass the original step arguments to the prefix. Construct tail
arguments from `result.routes`: `prefix_output` uses the computed output tuple;
`user_input` uses the existing original argument references directly. In an
iterative model these include the existing sample and device index carry. Keep
the original cache dependencies, loop, scheduler and output bindings unchanged.

The new Region boundary can add backend launch and output-copy costs. Test an
unmodified original, unchanged recompiled control, partitioned ATen baseline
and partitioned candidate independently. Preserve complete output checks and
source/artifact lineage before measuring end-to-end performance. Kernel timing
does not establish an end-to-end speedup, and this transform is not selected for
deployment merely because construction or CPU tests succeed.
