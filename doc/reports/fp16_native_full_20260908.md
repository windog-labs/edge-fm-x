# FP16 Native Full-Model Validation

2026-09-08 23:45 UTC. The FP16 head replacement has now completed a real
full-model no-Python Session validation on the local RTX 3060. All 16 complete
action chunks match the FP16 Python free-running candidates byte-for-byte and
all 16, including the eight held-out episodes, pass the original quality gate.

## Pipeline

- Shared original Regions were traced together with the rewritten FP16 step
  Region using the public TorchScript exporter. Four artifacts were produced:
  `vf_cached_smolvla_fresh_prefix_51`, `smolvla_fresh_initialize`,
  `fp16-step` and `smolvla_fresh_finish`.
- The public full-model bundle builder produced one quantized-lane C++ bundle
  with provider-required numerical bindings and no Python linkage.
- A single no-Python native Session consumed all 16 real inputs and emitted 16
  complete FP32 action chunks. Process maps show LibTorch but no Python runtime.

## Result

- Native action bytes equal the FP16 Python free-running candidate bytes for
  all 16 samples.
- `native-report.json` status `candidate_exact`, official quality gate
  `passed`, 16/16 rows pass, held-out 8/8 pass.
- Independent audit SHA:
  `35c660042d1785847ef2c98a3b529876446830118108d75b95e22022c827af85`.

## Evidence Hashes

- Protocol: `c5e34be3e5890d28322ce5a835e8d22fd9e5fdf3f4f2c755035ba1ee424d43c1`.
- Trace report: `4b91b95ffe077a653572d7ae2b50ef289854fd31051de8249ee438218afe52e5`.
- Build report: `eeec953aa1d662bf10b33faf35f0af13d950d1eb32cf7def2bdafc779718b370`.
- Native report: `873bc3fb92515f2edcf0a9f0905fdcc6eaa92d824e2eb6ae97d329e0d1c43e2e`.
- Execution protocol: `ed2342bdd1011aa742001434ad0de0447c72e5b94d2209a795f0e720e38e6f12`.
- Native PID 2077538, exit 0, final GPU owner list empty.

The first build attempt used a `low-precision` numerical lane before the public
contract accepted only `same-precision`/`quantized`; the failed receipt is
preserved and the successful run uses `quantized` lane metadata with
`candidate_precision=float16`.

## Remaining Boundary

This is no-Python deployment correctness, not latency, memory, replay, BF16
native, formal CDF, physical action or lossless acceptance. The execution lane
metadata remains limited by the public numerical contract; half-native behavior
is not a claim that all paper text can call this "quantized INT8". Orin/BPU and
full-paper acceptance remain deferred.
