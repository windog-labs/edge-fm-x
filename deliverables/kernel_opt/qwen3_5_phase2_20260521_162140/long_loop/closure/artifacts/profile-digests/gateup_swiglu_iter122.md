### Profile Evidence Digest: `edgefm_decode_swiglu_warp_kernel<16>` @ Iter122

Environment
- GPU: RTX 3060 / SM86.
- Shapes/dtypes: Qwen3.5-2B decode, batch rows `1`, hidden `2048`, intermediate `6144`, BF16 weights/activations.
- ncu report: `deliverables/kernel_opt/qwen3_5_phase2_20260521_162140/profiles/ncu/qwen3_5_2b_decode_swiglu_warp_iter122.ncu-rep`
- ncu csv: `deliverables/kernel_opt/qwen3_5_phase2_20260521_162140/profiles/ncu/qwen3_5_2b_decode_swiglu_warp_iter122_details.csv`
- nsys report: `deliverables/kernel_opt/qwen3_5_phase2_20260521_162140/profiles/qwen3_5_nsys_action_iter121.md`

Headline
- Bottleneck class: Memory-bound.
- Most-stalled reason: low eligible warps while DRAM is saturated; this is expected for skinny-BF16 GEMV.
- Confidence: High. DRAM and memory throughput are both above `95%`, and occupancy is above `93%`.

Evidence
- Duration: `153.70 us` per first matched launch.
- DRAM Throughput: `95.34%` -> at documented memory ceiling for this phase.
- Memory Throughput: `95.34%` -> same ceiling as DRAM, no separate L2-only bottleneck.
- Compute (SM) Throughput: `23.99%` -> expected for memory-bound GEMV.
- Achieved Occupancy: `93.18%`, theoretical `100%` -> occupancy is not the primary limiter.
- Active Warps Per Scheduler: `11.12`; Eligible Warps Per Scheduler: `0.34` -> many active warps are waiting on memory, consistent with DRAM-bound behavior.
- Block/Grid: `512/384`, registers/thread `33` -> enough parallelism to saturate memory on SM86 for this shape.

Hypotheses

1. Current exact BF16 GEMV+SwiGLU is at ceiling.
   - Why: DRAM `95.34%`; pure launch retuning can only recover a small residual.
   - Action: Do not keep retuning block/warp shape in production.
   - Expected impact: at most low single-digit percent of this kernel, below high-value threshold after graph/runtime noise.
   - Risk / cost: compile churn and token-risky numerical changes for marginal gain.

2. Reduce bytes through compression/quantization.
   - Why: memory is the bottleneck and the work reads two BF16 weight rows per output element.
   - Action: future separate project for FP8/INT8/INT4 weight-only or block-scaled GEMV with explicit scale/tolerance contract.
   - Expected impact: material if the token contract allows lower precision.
   - Risk / cost: changes correctness contract; must revalidate token alignment or define acceptance tolerance.

3. Larger fused projection design.
   - Why: a single GEMV is memory-bound, but a wider fused chain may reuse hidden state and scheduling better.
   - Action: standalone long-loop only, then migrate if full-generate improves.
   - Expected impact: uncertain; requires custom design beyond this phase.
   - Risk / cost: substantial code and routing complexity.

Next concrete edit
- File: none in production.
- Change: close the exact-BF16 local tuning loop and commit the accepted Iter121/Iter122 state.
- Validation: keep current Qwen3.5 fresh-dump and Qwen2.5 guard results as the closure gate; any future work starts from a new standalone algorithm spec.
