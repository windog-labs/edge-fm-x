# Trace-Bound Activation Observation

`analysis.precision_probe` is an offline diagnostic, not a quantized executor or
a timing path. It executes saved `ExportedProgram` operations through the public
FX Interpreter and returns the original output structure. It never replaces an
operator, adds graph outputs, changes weights, or selects a policy by model name.

## Interface

`ProbeRegion` binds a saved EP path, SHA256, region name, and explicit context or
iteration stage. `CalibrationSite` binds a top-level Tensor-producing
`call_function` node to that exact artifact. Nested HOP internals, dynamic input
profiles, custom tensor subclasses, and non-Tensor selected results are rejected.
Nested HOP execution itself remains original and receives the recursive effect
audit. Hidden RNG, external I/O, and external/persistent mutation fail before use.

`PrecisionProbe` accepts the full profile document and current versioned
`NumericalContext`, computes canonical digests internally, and checks the current
numerical context at sample entry and exit without setters. The process lease
does not claim to stop external PyTorch global-policy writes.

`observe_sample` validates actual named observation and noise tensor identities,
executes the supplied complete invocation, and requires exactly one call per
context region and all declared steps per iterative region. Call records include
actual input fingerprints and step ordinals. Step labels and full scheduler/CFG
semantics belong to the frozen profile and source graph; they are not inferred
from model names.

After each selected node returns, its value is copied into newly allocated CPU
storage before any subsequent in-place operation. The explicit `empty(shape,
dtype, device=cpu).copy_(detach)` is intentional: PyTorch considers singleton
views with non-unit stride contiguous, but they cannot be directly reinterpreted
as byte storage. The actual execution tensor is not replaced by this copy.

## Acceptance

The reference output tree and complete dtype/shape/byte identities are frozen
before invocation; references cannot mutate during the callback. Every original
output must match the frozen reference and floating outputs must be finite.
External input values and layout are checked unchanged. CPU and participating
CUDA implicit RNG states are checked unchanged.

Persistent parameters, buffers, and Tensor-valued graph constants are audited
by actual contents plus layout/storage identities at trajectory entry and exit.
Version checks additionally guard region boundaries, but are not used as a
substitute for content checks: `.data.add_` and `.data=` can evade versions.
This full content audit deliberately adds synchronization and host copies.

Only a completed `ProbeResult` exposes observations. `publish(collector)` returns
a separately staged collector, so a failure cannot partially change the original.
The sample and all profile/site/step/context declarations must match. Held-out
samples can run through the same output gate but cannot publish calibration
observations. `owned_snapshot(site, step)` returns a new independent CPU clone
for explicitly retained sites, not a writable view of internal evidence.

## Generic Tool

`tools/probe_precision.py` accepts strict
`vlaforge.precision_probe_protocol/1` JSON. It binds the original source Module
file, every Region artifact, full input and output files, profile, schedule,
versioned numerical context, named sites, fixed sample split, and explicit
`retain_sites`. Missing or unknown fields, duplicate JSON keys, missing outputs,
bad hashes, or wrong dtypes/shapes fail closed.

The current tool constructs the real IR `Interpreter` with the public
`InvocationProgram` predicate validator and reads every public output after run.
It supports stateless invocations and contiguous externally bound tensors.
It does not implement model-specific loops or arbitrary custom validators.
Its report records the source Module hash and actual invocation name; the helper
alone labels its callback caller-supplied and is not an independent IR proof.

Each complete output is archived as original-dtype raw bytes. Only explicitly
retained activation sites are stored in full, with sample, split, step, shape,
dtype, SHA256 and source Region identity. All other sites retain statistics.
Four scale plans are derived only from the calibration partition. These plans
and observations do not demonstrate a deployed low-precision kernel, acceptable
quantized model output, memory reduction, or speedup.

Actual GPU execution requires an external exclusive-device owner supervisor.
The tool's declaration is not itself an ownership lock. The recorded SmolVLA
experiment supplies that supervisor and terminates only its own child if another
compute owner appears. Graphics owners are never terminated.
