# Explicit Numerical Workers in Session Benchmarks

The benchmark tool preserves the Session's verify-only numerical contract.
It never silently configures process-global policies. Preparing any bundle
with numerical bindings requires an explicit protocol declaration:

```json
{
  "numerical_worker_bootstrap": {
    "mode": "libtorch-explicit-exclusive-process/1",
    "acknowledge_exclusive_process": true,
    "acknowledge_calling_thread": true
  }
}
```

Omission preserves legacy unbound preparation without setters. A policy-bearing
bundle without the declaration is rejected before preparation writes output or
builds C++. An enabled declaration requires every Region to have a typed,
runtime-deployable `provider-required/1` binding. Legacy gaps, unimplemented
enforcement, incompatible policies, unsupported namespaces/backends or policy
versions are rejected. The common LibTorch policy contains 22 math settings
and two release/API identity fields, not 24 independent math settings.

The tool reuses `generate_libtorch_worker_initializer`. In the generated runner
the checked call occurs on the calling thread, after any optional CUDA owner
registration/reset/re-registration handshake and before input allocations or
Session creation. Its template location is a C++ comment marker so older
renderers remain syntactically valid. No initialization happens from Session
construction or run, and no CUDA reset is added to the model lifetime.

Successful initialization emits one policy-digest marker before timing begins.
Preparation freezes the initializer sources, per-Region binding digests and
initialization mode. Each process report verifies the marker against those
frozen declarations; aggregation repeats that check. Missing or duplicate
markers, changed metadata, or absent source bindings fail closed. The marker
is preparation evidence, not a replacement for Session checks before load,
at run entry and at completion before commit.

The acknowledgements declare ownership; they cannot prevent unrelated code
from invoking global setters. Separate incompatible workers remain separate
processes. Ordinary latency, allocator measurements, complete output bytes and
official quality gates retain their existing boundaries and validations.
