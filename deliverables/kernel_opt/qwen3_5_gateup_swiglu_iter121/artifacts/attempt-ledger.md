# Attempt Ledger

## Iter121-A: one-warp-per-intermediate GateUp+SwiGLU

- Intent: test a simple EdgeFM-owned decode fused projection/activation kernel before considering any production migration.
- Shapes: Qwen3.5 0.8B hidden 1024 / intermediate 3584; Qwen3.5 2B hidden 2048 / intermediate 6144.
- Variants: scalar and BF16x2 loads, `8/16/24/32` warps per block.
- Status: harness added; benchmark pending.
