# Optimization Ledger

No standalone candidate has been accepted yet. Production migration is gated on exact top1 agreement and a clear latency win on both Qwen3.5 LMHead shapes.

## Rejected

- `bf162` vectorized row-dot load path: exact on random standalone 0.8B/2B shapes, but only `~0.2%` faster than production-equivalent scalar24. This is below the migration threshold and consistent with the prior NCU conclusion that LMHead top1 is already DRAM-roofline limited.
