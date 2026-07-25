# VLAForge Invocation IR v0.2 Real-Model Evidence

Date: 2026-07-25

These reports are real checkpoint eager-to-Invocation-IR L2 evidence. They are
not real compiled artifact L3 or generated no-Python C++ L4 evidence.

## SmolVLA

- checkpoint SHA-256:
  `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`
- LeRobot revision: `8fff0fde`
- device: NVIDIA GeForce RTX 3060
- 10 solver-step maximum absolute errors: all `0.0`
- final action maximum absolute error: `0.0`
- three action-queue maximum absolute errors: all `0.0`
- schema: `vlaforge.real_model_evidence/0.2`
- result: passed

Artifacts:

- `smolvla_eager_ir.json`
- `smolvla_trace.json`

## OpenVLA

- checkpoint revision:
  `47a0ec7fc4ec123775a391911046cf33cf9ed83f`
- three checkpoint shard hashes are recorded in the JSON report
- quantization: bitsandbytes NF4
- device: NVIDIA GeForce RTX 3060
- generated action token IDs: exact
- action maximum absolute error: `0.0`
- schema: `vlaforge.real_model_evidence/0.2`
- result: passed

Artifacts:

- `openvla_eager_ir.json`
- `openvla_trace.json`

The recorded peak-memory and single-run timing fields are audit metadata, not
paper-grade latency distributions. Use a separate warmed, repeated benchmark
before making performance claims.
