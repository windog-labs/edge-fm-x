# Attempt Ledger

## Closure Pass 2026-05-23

- Intent: decide whether to continue Qwen3.5 long-loop optimization after local P0-P4 kernel work.
- Inputs:
  - Iter121 accepted GateUp+SwiGLU GEMV graph-on gains.
  - Iter122 NCU for accepted GateUp+SwiGLU GEMV.
  - Existing LMHead/P0/dense-linear blockers in `CURRENT.md` and `state.json`.
  - KernelPilot knowledge pass for GEMM, activation fusion, quantization, TensorRT-LLM, vLLM, and GEMM/quant PRs.
- Result:
  - No new production candidate was opened.
  - Remaining material gains require changing bytes/precision/algorithm or runtime scheduling.
  - Current exact-BF16 local loop is closed as low ROI.
- Decision: converge, commit, and push current accepted state.
