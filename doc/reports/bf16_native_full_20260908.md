# BF16 Native Full-Model Validation

2026-09-08 23:55 UTC. The BF16 head replacement completed the same no-Python
full-model Session validation as FP16. All 16 complete actions match the BF16
Python free-running candidates byte-for-byte and all 16, including eight
held-out episodes, pass the original quality gate.

The shared Regions plus BF16 step Region were traced with the public exporter,
built into one C++ bundle with provider-required bindings, and executed by one
no-Python Session on RTX 3060. Process maps contain LibTorch but no Python
runtime.

## Hashes

- Protocol: `6a6e7aef96bd846fd6d854067290f75a0b69251f87b931c3015836a7a336d96d`.
- Trace report: `71f98150707c29108c6fb30815911bdf0f5a09597035f5cacf732fc5230d2e26`.
- Build report: `2332246f59df50bb3b73c91107b8ad7c029fcf04e3a11ab04d9eb9e52d91e631`.
- Native report: `9fcb7974b85f73e95cf4b0cb1443355828b131d3feea24b993bb7f95c55315ef`.
- Independent native audit:
  `0002d4d3ce0b3f99db79ad6cb63f7302ffc4a9c1033fc78cfff21aa097cfc7f9`.

## Remaining Boundary

This is no-Python deployment correctness, not latency, memory, replay, formal
CDF, physical action or lossless acceptance. BF16 held-out worst Python MSE is
5.109796e-6; native outputs are byte-exact to that Python lane. Orin/BPU and
full-paper acceptance remain deferred.
