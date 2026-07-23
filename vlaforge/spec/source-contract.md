# VLAForge Restricted Python Source Contract

VLAForge does not attempt to recover robot semantics from arbitrary Python.
The source frontend accepts explicitly annotated policies with the following
contract.

## Tensor regions

- Inputs and outputs have declared IR types.
- The callable is deterministic for the same explicit inputs.
- Persistent buffers and RNG are explicit inputs and outputs.
- No file, network, robot, global-state, or hidden cache effects occur.
- Dynamic shapes are represented by declared shape dimensions and guards.

The `@tensor_region` annotation records this contract. A later
`torch.export` frontend will audit the captured `ExportedProgram` against it.

## Program semantics

The author declares:

- policy trigger clock;
- input-stream clock and maximum staleness;
- persistent state ownership, version clock, retention, reset, and freshness;
- loop bounds or maximum iterations;
- action validation and commit point;
- asynchronous state read/write sets.

The frontend may infer tensor types and SSA dependencies. It must not infer
control frequency, action visibility, reset policy, acceptable staleness, or
whether an approximation is safe.

## Adapter boundary

Model adapters may:

- map model input dictionaries to typed input streams;
- split a source model into pure regions;
- expose source RNG/cache as explicit state;
- register Python callables for the reference interpreter;
- record eager/reference traces.

Adapters may not:

- add model-named core operations;
- mutate the generic interpreter;
- silently hide unsupported state inside a region;
- label deterministic fixtures as real-checkpoint evidence.

