# Paper Claims and H20 Evidence

This review checks the seven-page `doc/ICRA_2027_EdgeFM.pdf`, SHA256
`92a3af4618a08c7ea66e62b928c3bb32d326e0445a491eee079ecccf70adf997`.
The PDF text was extracted and pages 4-5 were rendered and visually inspected.
The original PDF is unchanged. The author's newer request explicitly lists
five VLAs, including separate pi0 and pi0.5 experiments; that list takes
precedence over the draft's four-model wording on page 5.

The current task covers H20 experiments on `zzm-h20-x8-1` and `zzm-h20-x8-2`.
Orin and Journey/J6M execution is deferred by the user. Completion of this H20
task must remain distinct from acceptance of the whole edge-hardware paper.
The live model completion states are in `../edgefm_vla_goal_status.md`.

## Required Interpretation

| PDF Location | Claim or Draft Content | What the Current Evidence Supports |
|---|---|---|
| Pages 1-2, 5: edge and vendor performance | Orin speedups over TensorRT-Edge-LLM; Journey end-to-end deployment | The new experiments compare H20 paths. They do not validate Orin, BPU, power modes or a vendor-runtime speedup. Existing vendor/board results need their own evidence. |
| Page 5, IV-B: model list | Four VLAs, with pi0.5 only represented by a citation and CogACT labeled 3B | Report five separate models. The actual CogACT checkpoint has 7,630,224,071 parameter elements and uses the declared public dependency profile. A 3B complete-model label would be incorrect. |
| Pages 3-5: inference time and control frequency | Per-control-cycle latency and action-generation Hz | The complete host pipeline times decoded CPU inputs through complete CPU action chunks. Throughput means fresh action chunks/s. It does not equal robot servo frequency or measured closed-loop responsiveness. |
| Page 4: zero-copy sensorimotor pipeline | Camera buffers mapped directly to inference and action execution | H20 host measurements explicitly include H2D and D2H. Resident-tensor measurements exclude these transfers. Neither establishes a board camera/actuator zero-copy pipeline. |
| Page 5: timing guarantees | Jitter elimination and a standard deviation below 1.2 ms | The CDFs and raw samples establish empirical distributions for the recorded profiles. They do not prove a hard deadline, zero jitter or a universal 1.2 ms bound. For example, raw-input required RDT std is 3.366552 ms and pi0 std is 1.901058 ms. |
| Pages 3-4, 6: numerical equivalence | Exact execution and output-equivalent aggressive low precision | The new formal quality checks compare complete normalized/native typed outputs at the recorded numerical settings. They do not establish arbitrary INT8/INT4 equivalence, stepwise quantization-scale correctness or lossy-weight equivalence. |
| Page 6: fidelity and trajectory count | A thousand trajectory steps and preserved policy behavior | Formal calls cycle through the finite recorded observation/noise sets. Repeated measurements must not be counted as distinct trajectories, episodes, independent tasks or physical success-rate tests. Exact outputs support equality on the tested inputs. |
| Pages 4-6: agent workflow and operator table | Automatic synthesis, universal operator speedups and weeks-to-minutes onboarding | The six-family H20 table supports agent-assisted selection of fixed compiler recipes on archived signatures. Attention is unchanged and Linear is slower. It does not measure onboarding effort, autonomous kernel invention, skill reuse across hardware or integrated-model gains from these candidates. |
| Page 6: VLM generalization | Native VLM compatibility and vendor comparisons | Qwen3.5 0.8B/2B now have fixed-profile H20 native deployment for first-token and 16-token generation, complete output parity, CDFs and Session lifecycle audits. The earlier cache-copy failures remain historical route evidence. This does not establish arbitrary prompt/image/generation support or Orin/J6M/BPU compatibility. |

## Reporting Boundaries

Keep three VLA timing tables distinct:

1. CUDA-resident model tensors through Session completion, preserving the
   off, batch-only and required policy identities.
2. Host model tensors through H2D, binding, Session completion and all-output
   D2H. This excludes raw-image preprocessing.
3. Decoded raw observations in host RAM through preprocessing, inference and
   complete CPU action outputs, with qualified official-component and generated
   Session paths. This excludes file/video decoding, initialization, logging,
   reference comparison and robot transport. It uses a Python host.

The original-input timer stores two measured segments whose sum equals each
complete-call sample: preprocessing plus H2D, and inference plus postprocessing
plus D2H. It does not independently attribute each operation inside those
segments. Adapter initialization and sampled process memory are reported
separately. The memory counter is not a proof of the paper's embedded on-chip
memory budget or absence of all dynamic allocations.

Every formal configuration uses at least five independent processes, each with
128 warmups and 1024 measured calls. Warmups remain in output-fidelity storage
but are excluded from latency statistics. The empirical CDF retains all measured
outliers. No deadline-miss fraction is implied without an explicit robot control
period and action-execution window.

## Model Qualifications

- SmolVLA uses 16 recorded observations from 16 SO100 episodes and the documented
  recovered statistics. Native output scale does not establish robot calibration.
- RDT uses recorded AgileX observations, six historical camera views, online
  T5/SigLIP, the original scheduler and preserved BGR-to-PIL profile. The official
  components run in the named Torch 2.10 environment. Complete native actions
  are BF16 before the wrapper's lossless F32 cast.
- pi0 and pi0.5 use 16 frames from one ALOHA episode, with saved noise and
  complete normalized/native outputs. They are not sixteen-episode evaluations.
- CogACT uses 16 spaced frames from one Fractal episode. The autonomous C++
  `CudaRngProvider` path is verified against the official RNG sequence and has a
  separate same-boundary formal comparison. Original Meta configuration
  equivalence remains unverified, and the raw-input campaign's archived external
  RNG tape remains part of that separate boundary.
- Qwen3.5 formal baselines each use one real 640x480 image/prompt, 327 input
  tokens including 300 image tokens, and 16 generated tokens. The native
  deployment covers Qwen3.5-0.8B and 2B for first-token and fixed-16-token
  profiles, with complete output byte equality and generic Session lifecycle
  probes. These repeated runs do not expand input coverage or establish
  arbitrary streaming behavior.

## Source Reports

The model-tensor tables are `h20_vla_formal_table_20260909.md` and
`h20_host_tensor_formal_table_20260909.md`. Original-input results and status
are in `raw_input_host_pipeline_20260910.md`,
`raw_smolvla_rdt_pipeline_20260910.md` and
`raw_cogact_host_pipeline_20260910.md`. The exact operator scope and negative
results are in `h20_operator_table_20260909.md`. VLM data and native results are
in `qwen35_natural_profile_20260909.md` and
`qwen35_native_formal_20260910.md`; the earlier rejected routes remain in
`qwen35_native_blockers_20260910.md`. The boundary-aligned CogACT result and
historical model-only comparison are in
`cogact_timing_boundary_formal_20260910.md` and
`cogact_autonomous_native_formal_20260910.md`.

The draft also has unresolved figure/section references, refers to Figure 1
as a latency CDF while its caption describes the optimization workflow, and
mixes older Octo/OpenVLA table content with the newer VLA list. Those editorial
items require reconciliation when incorporating the new results; this task does
not rewrite or regenerate the PDF.
