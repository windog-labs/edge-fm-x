# Attempt Ledger

## Iter118-A: BF16 LMHead Top1 Standalone Sweep

- Intent: test whether production-equivalent scalar top1 can be improved by BF16x2 loads or a different one-warp-per-row occupancy point before touching `src/layers/linear.cu`.
- Shapes: Qwen3.5 0.8B `vocab=248320, hidden=1024`; Qwen3.5 2B `vocab=248320, hidden=2048`.
- Correctness anchor: scalar 24-warps-per-block token plus optional `torch.mv(...).argmax()` reference.
- Variants: scalar `16/24/32` warps per block, BF16x2 `16/24/32` warps per block.
- Result: full 0.8B/2B sweep passed token agreement against scalar24 and `torch.mv(...).argmax()`.
- Best standalone variants:
  - 0.8B: `bf162_w24` `1.4970 ms` vs scalar24 `1.4995 ms` (`~0.2%` faster).
  - 2B: `bf162_w32` `2.9786 ms` vs scalar24 `2.9854 ms` (`~0.2%` faster).
- Artifact: `benchmarks/lmhead_top1_iter118_baseline.json`.
- Status: rejected for production migration. The speedup is too small to justify touching the production LMHead path or running a full token-alignment/benchmark transfer.
