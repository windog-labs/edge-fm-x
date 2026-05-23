# Qwen3.5 GateUp + SwiGLU Iter121

Standalone feasibility check for a Qwen3.5 decode `gate_up_fused + SiLU * up` custom GEMV.

The accepted production path uses cuBLAS/cuBLASLt for the fused `[up, gate]` projection and a separate activation kernel. The existing TRT-LLM fused-MoE decode path is not safe for Qwen3.5 2B on SM86, so this directory tests a simpler EdgeFM-owned one-warp-per-intermediate candidate before any production migration.

Acceptance for migration would require:

- output closeness against the two-stage BF16 reference on 0.8B and 2B shapes;
- standalone latency clearly better than the current measured library tactic plus activation chain;
- Qwen3.5 0.8B/2B fresh-dump token alignment after migration;
- graph-on benchmark improvement on affected cases.
