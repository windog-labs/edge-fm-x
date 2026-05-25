# Qwen3.5 LMHead Top1 Iter118

Standalone exploration for exact greedy `lm_head_top1` alternatives.

The in-repo production path is already DRAM-roofline on RTX 3060, so this
directory is intentionally isolated. A candidate is not eligible for migration
unless it is exact, faster on both Qwen3.5 0.8B and 2B LMHead shapes, and later
passes full EdgeFM token alignment plus generate benchmarks.

## Shapes

- 0.8B: vocab `248320`, hidden `1024`, BF16
- 2B: vocab `248320`, hidden `2048`, BF16

## Acceptance

- Exact argmax token and tie-break match the reference.
- Candidate median latency improves over the scalar production-like baseline.
- Any production transfer must run the full Qwen3.5 fresh-dump and affected
  graph-on benchmark gates.
