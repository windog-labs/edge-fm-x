# Qwen Gating Operator Diagnosis

## Completed Localization And Repair

All four original cases and four intermediate-output cases are terminal,
with manager/program states unchanged. Input echo and parameter Cast are
byteexact; Softplus is near-reference, but static parameter Exp first diverges.
Stocktrtexec reproduces both old same-engine errors. Selective readonly
Cast/Exp folding restores all4source branch outputs, finite/maxabs5.96e-8.
This is operator evidence, not whole-model acceptance or performance.
The generic helper has111related tests and complete15step/49output CPU
decoder parity; target reintegration follows. [Current repair evidence](orin_qwen_readonly_exp_repair_20261025.md).

## Current Boundary

2026-10-25 08:59:53 UTC: the corrected no-service operator suite is verified
live, PID3700169, owned unit `edgefm-qwen-build-native-gate-probes-20261025-15-002.service`.
Three of four independent cases are collected. This is diagnosis, not model
acceptance or operator/E2E performance data. Five accepted VLAs are unchanged.

Two frozen branches, each9original nodes, are extracted from the first
recurrent boundary and the first nonfinite boundary. Inputs are real CPU
source activations produced from the archived actual Orin prefix. Appending
observers leaves all49complete parent outputs byte-exact; extracted outputs
are byte-exact to the parent. Original NodeProtos and learned inline A_log
payloads are preserved. Empty external sidecars mean parameters are inline,
not zero/fabricated weights. Each branch is tested under the two already
declared13/14mixed-precision policies, not a new precision search.

The first9node real Orin case under policy13 already reproduces a large
local error: output mostly0/1 instead of the reference damping factors,
MSE0.2021835744380951/maxabs0.9996435046195984. The complete model, other
state layers and prefix approximation are therefore not required for this
specific error. This does not prove the entire NaN cause yet.

## Tooling Failure And Qualified Resume

The first suite actually built the first two tiny engines. After the first
inference, the next metadata probe selected CUDA1 because the script confused
case index with the inspector's CUDA `ordinal`. Inspector source lines112-119
confirm that meaning. It was a coordinator bug, not unavailable hardware or
a model unsupported result. That unit is terminal/empty; manager/program
states unchanged. A prior wrong host import was also rejected before GPU
work or overwrite; phase-local transport now has an unambiguous module name.

The fixed worker takes a declared CUDA0 independently of case order;10device
contract tests pass. A fresh immutable resume manifest reuses both rehashed
engines and builds only the two missing cases. Source graphs, profiles and
real inputs are unchanged.600second cgroup limit and16GiB/4GiB admission/
floor guards remain, no service control or privileged operations.

## Evidence

- `artifacts/orin/20261025-15/preparation-002.json`
- `artifacts/orin/20261025-15/preparation-worker-002/attempt.json`
- `artifacts/orin/20261025-15/suite-assets-002/staging.json`
- `artifacts/orin/20261025-15/native-gate-probes-20261025-15-001/terminal-audit-001.json`
- `artifacts/orin/20261025-15/operator-device-contract-tests-001.xml`
- `artifacts/orin/20261025-15/suite-resume-assets-001/staging.json`
- `artifacts/orin/20261025-15/native-gate-probes-20261025-15-002/checkpoint-001.json`
