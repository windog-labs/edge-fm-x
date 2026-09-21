# Generic Python Invocation Interface

`InvocationBuilder` lowers typed Python orchestration to the existing
Invocation IR. It does not add model-specific opcodes or a second runtime.

```text
Python Adapter: typed ports, pure stages, bounded iteration, state semantics
  -> Invocation IR: SSA, inferred dependencies, validation and transaction
  -> compile_module: verified Plan, cache policy, arena and state storage
  -> generated C++ Session: binding, execution, commit/abort and reset

Pure tensor stages -> capture_region / torch.export -> backend artifacts
  -> existing RegionExecutable ABI used by the same generated Session
```

The Adapter supplies computation and semantic declarations. It does not assign
physical buffers, construct cache keys, advance state versions, or implement
transaction management in C++.

## Interface

Import `InvocationBuilder`, `InvocationProgram`, `SymbolicValue`, and
`tensor_region` from `vlaforge.frontend`.

| Method | Adapter declaration | Framework implementation |
|---|---|---|
| constructor | typed inputs/outputs and optional `StateSlot`s | schema and stable IDs |
| `input(name)` | external value needed by stages | one input read and revision provenance |
| `state(name)` | authoritative state needed this Run | one immutable snapshot and version provenance |
| `call(stage, *values, cache=False)` | pure annotated computation, optional exact reuse | SSA, type checks and transitive dependency inference |
| `iterate(initial, body, steps=N)` | fixed profile and step recurrence | one traced body, typed variadic `vla.for` |
| `update_state(name, value)` | next authoritative value | deferred write in the invocation transaction |
| `finish(outputs, accepted=predicate)` | named outputs and acceptance condition | validation, atomic state/output commit and return |

`call` and `iterate` always return tuples of symbolic results. A stage with one
declared output returns that value itself when executed, not an extra wrapper
tuple. `finish` returns an `InvocationProgram` containing a verified module,
reference stage implementations, `validators`, and `cpp_validators()`.
The invocation name is `act`. Optional `finish(metadata=...)` records Adapter
and evidence metadata without changing execution semantics.

`iterate(..., replay="prefer" | "required" | "batch-only" | "off")` optionally requests
[whole-loop CUDA replay](bounded-replay.md). The default is `off`; supported
loops get compiler-owned reset storage and generated callbacks. Static
eligibility, actual backend capture and numerical validation remain separate.
`batch-only` uses the same stable loop storage and generated callbacks, but
only queues ordinary work on the shared stream without graph capture. It is
an explicit ablation mode, not a claim of replay. Deployment can instead pass
`loop_execution=...` to `compile_module` or `build_artifact_compile_bundle`;
the default `source` preserves each loop's source declaration. An override
retains the original IR and records the selected policy in bundle metadata.

## Continuous Generation Example

This small tensor example is an interface demonstration, not a pretrained
policy or a performance benchmark. Shape, dtype, and step count are a static
profile supplied by the Adapter.

```python
from vlaforge.frontend import InvocationBuilder, tensor_region
from vlaforge.ir.program import InputPort, OutputPort, Value
from vlaforge.ir.types import TensorType

vector = TensorType((2,), "f32")
time = TensorType((1,), "f32")
flag = TensorType((), "bool")

@tensor_region("prefill", inputs=(Value("observation", vector),), outputs=(vector,))
def prefill(observation):
    return observation * 2

@tensor_region(
    "step",
    inputs=(Value("context", vector), Value("sample", vector), Value("time", time)),
    outputs=(vector, time),
)
def step(context, sample, time):
    return sample - 0.25 * (context + sample * time), time - 0.25

@tensor_region("accept", inputs=(Value("sample", vector),), outputs=(flag,))
def accept(sample):
    return sample.isfinite().all()

b = InvocationBuilder(
    "continuous",
    inputs=(InputPort("observation", vector), InputPort("noise", vector),
            InputPort("time", time)),
    outputs=(OutputPort("action", vector),),
)
observation, noise, start_time = b.input("observation"), b.input("noise"), b.input("time")
context, = b.call(prefill, observation, cache=True)
action, _ = b.iterate(
    (noise, start_time),
    lambda index, sample, time: b.call(step, context, sample, time),
    steps=4,
)
accepted, = b.call(accept, action)
program = b.finish({"action": action}, accepted=accepted)
```

The step callback executes once at construction. Its IR executes four times.
Use tensor-valued time/schedule inputs for exported tensor stages when their
values must vary at runtime; an ordinary Python integer passed to
`torch.export` can be specialized. The symbolic induction value is an IR
`ScalarType("index")`, not a PyTorch tensor.

For autoregressive generation, replace `(sample, time)` with typed
`(token, KV, position, ...)` carry and provide different pure stages. The core
requires no model-family switch. The two executable deterministic examples in
`examples/iterative_frontend.py` exercise continuous and autoregressive
generation. The existing `adapters/pi0/pi0.py` L1 fixture uses this same interface.

## State and Context

Three distinct lifetimes remain explicit in Python semantics:

- Persistent authoritative values use declared `StateSlot`s and
  `state`/`update_state`. Supply initial values through the existing
  interpreter or codegen `initial_state` interface (or Session initialization).
  Reset policy belongs to the slot declaration; the Session implements it.
- Derived prefill/context values are normal stage results. Place their call
  before `iterate` to compute once per Run; use `cache=True` to request exact
  cross-Run reuse under the selected compiler profile.
- Invocation-local samples, token/KV values and solver state use loop carry.
  They are not persistent merely because generation has multiple steps.

Cache input/state dependencies are inferred transitively from symbolic values,
including captured context and all loop-carried inputs. Each cached call site
has an independent Region identity. Model weights and other closed-over
configuration must be immutable for a compiled Session; mutable state and RNG
must be explicit values. No tensor address is used as input identity.

The caller may omit `InputStamp.revision`; the runtime then prevents unsafe
cross-Run reuse by assigning fresh automatic identities. To enable reuse, the
caller must supply the same explicit revision only for logically unchanged
data. Automatic revisions, explicit revisions and optional defaults occupy
distinct cache-key domains, even when their numeric values coincide.
Changed dependent state versions and
episode resets also invalidate context reuse. Unrelated ports are excluded.

The acceptance condition is one bool scalar or a singleton bool tensor.
Implement arbitrary tensor validity checks as pure stages returning that
predicate. `program.cpp_validators()` supplies the generic final predicate
validator. A failed predicate or execution does not publish staged state or
outputs; the previous committed output remains readable.

## Lowering and Memory

Compile `program.module` with the existing `compile_module` interface. Pass
`program.regions` and `program.validators` to the reference Interpreter or
PlanExecutor. For deployment, capture the declared Regions using their
implementations in `program.regions`, compile them with an existing backend,
and bind those artifacts through the existing codegen/bundle interface.
`InvocationBuilder` does not itself invoke the artifact compiler or download
weights. Inline `CppRegionDefinition`s in the tests are deterministic ABI
fixtures, not evidence of tensor artifact compilation.

`vla.for` accepts one or more heterogeneous carried values. Initial values,
body arguments after the induction argument, yielded values and results must
have matching arity and types. Updates are simultaneous, including swaps.
The Plan allocates typed `carry_scratch` buffers for multi-value loops; C++
stages all yields before overwriting any carry. Scratch participates in the
static arena, with no per-iteration heap allocation for this staging.

Values defined outside a loop and used inside it remain live through the whole
loop body, not merely through their last textual use. The same lifetime rule
applies to nested loops. Loop results and carry scratch are pinned while the
loop executes. The single-carry IR representation remains supported; the
variadic representation adds `iter_args` and Plan `carry_scratch` attributes,
not a new opcode. Rebuild the Plan and generated Session together when adopting
this extension; older compilers need not accept variadic loops.

## Limits and Evidence

- This is a restricted symbolic Interface, not arbitrary Python-to-C++
  translation or automatic model graph partitioning. Branching on a
  `SymbolicValue` is rejected. Structured `vla.if` remains available through
  the lower-level IR; this builder currently offers only bounded iteration.
- `steps` is a positive compile-time integer. Loop carries have static types;
  changing KV shapes, early termination and dynamic profiles require explicit
  compatible representations or further frontend support.
- Inputs and persistent state are declared/read outside loops. Loop-local
  values cannot escape except through the returned carry. A failed callback
  invalidates the builder. State writes remain pending until final commit.
- Prefill is explicitly placed before the loop. This change is not a general
  LICM pass that discovers arbitrary invariant computation inside model code.
- Existing backend limits remain: no whole-loop CUDA Graph capture, no BPU
  provider, no multi-device unified arena, and no takeover of AOTI-owned
  internal workspace or weights in this change.

`tests/frontend/test_invocation.py` verifies both paradigms against the IR and
Plan, serialization, cache provenance, failure/reset semantics, nested loops,
memory lifetimes, torch-export parity and compiled no-Python CPU Sessions.
C++ coverage includes simultaneous swaps, uncached context reuse, revision
domain collisions, state rollback and episode reset. These are deterministic
fixture results, not real-checkpoint L3/L4 or edge-device performance evidence.
