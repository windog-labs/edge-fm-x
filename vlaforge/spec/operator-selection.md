# Operator Selection Evidence

`select_operator_candidate` is a model-independent metadata consistency gate,
not a verifier that a supplied JSON report describes a real execution. Producers
must independently verify compiled payloads, native execution, full outputs and
timings before passing their reports. Unit-test selection is not model evidence.

The returned `vlaforge.operator_selection/2` record binds canonical digests of
both complete reports. Canonical JSON uses sorted keys, compact separators,
finite numbers, and UTF-8; `candidate_report_sha256` hashes exactly the candidate
mapping passed to the Interface, including any derived microbenchmark summary.
Raw report/file hashes should also remain in that mapping's provenance.

The operator report requires its measured `gpu` type and `artifact_sha256`.
The complete-model report requires `operator_integration` with schema
`vlaforge.operator_integration/1`, `candidate_report_sha256`, and two lanes named
`baseline` and `candidate`. Each lane supplies:

- `gpu`: the actual GPU type from execution evidence, matching the microbenchmark.
- `bundle_sha256` and `execution_audit_sha256`: distinct per lane.
- `loaded_artifact_sha256`: digests of the actual verified bundle payloads.
- `input_contract_sha256`, `output_contract_sha256`, `numerical_policy_sha256`
  and `measurement_protocol_sha256`: identical between the two lanes.
- `measured_calls`: a positive integer matching the complete-model report.
- `mean_latency_ns`: a positive finite measured mean.

The candidate artifact must occur in the candidate lane and not in the baseline.
A reconstructed or fused artifact is not accepted as the same measured artifact;
it needs a future explicit derivation contract and new correctness evidence.
No model or GPU name controls dispatch. The `hardware` argument is a display
label, not the source of the GPU-type binding.

The reported speedup must agree with the two lane means. Selection additionally
requires complete-output byte equality, the candidate's microbenchmark numeric
gate, explicit `confidence_gate_passed=true`, and a strictly positive E2E gain
meeting the requested minimum. Missing bindings or confidence evidence reject
the candidate. A rejected candidate remains a reportable experiment outcome.

The original v1 interface accepted integration flags without identity binding.
Do not promote historical v1 records. In particular, a static-precompute
original/clone-control comparison is not evidence that an independently
compiled operator was inserted into a complete model, even when all reports
separately contain genuine measurements.
