# VLAForge Report Status

Deployment design: [J6P VLM with two complete HBM models](../j6p_vlm_deployment_and_kv_cache.md)
documents the Python example and explicit prefill/decode cache ABI. It is not
board execution evidence.

## Current H20 Delivery

Start with [the final H20 experiment delivery](h20_vla_experiment_delivery_20260910.md),
[Goal status](../edgefm_vla_goal_status.md) and
[the PDF claim review](h20_paper_scope_review_20260910.md).

The current H20 evidence deliverables are complete, including the Goal's allowed
Qwen native deployment for the fixed H20 profile. Full edge-paper acceptance is
not complete: Orin remains deferred, J6M coverage is limited to the SmolVLA
stage evidence below, and each model's documented profile limitations remain in force.

| Evidence | Final Report |
|---|---|
| Five VLA original-input official/Session table, complete action quality, CDF and evidence index | [Unified delivery](h20_vla_experiment_delivery_20260910.md) |
| pi0 and pi0.5 original-input formal campaigns | [OpenPI host pipelines](raw_input_host_pipeline_20260910.md) |
| SmolVLA and RDT original-input formal campaigns | [SmolVLA/RDT host pipelines](raw_smolvla_rdt_pipeline_20260910.md) |
| CogACT 10/10 formal workers, full audits and CDF | [Qualified CogACT host pipeline](raw_cogact_host_pipeline_20260910.md) |
| CogACT autonomous C++ RNG and historical model-only comparison | [Autonomous native formal result](cogact_autonomous_native_formal_20260910.md) |
| CogACT same-boundary official/native formal comparison and CDF | [Boundary-aligned comparison](cogact_timing_boundary_formal_20260910.md) |
| Five-model CUDA-resident tensor comparison | [Resident table](h20_vla_formal_table_20260909.md) |
| Five-model per-call H2D/Session/D2H comparison | [Host model-tensor table](h20_host_tensor_formal_table_20260909.md) |
| Six operator families, five process pairs each, including negative results | [Operator table](h20_operator_table_20260909.md) |
| Qwen3.5 0.8B/2B official real-image formal baselines | [Multimodal table](qwen35_natural_profile_20260909.md) |
| Qwen3.5 0.8B/2B native first-token and 16-token formal deployment | [Native formal result](qwen35_native_formal_20260910.md) |
| Historical rejected Qwen native export routes | [Native blocker report](qwen35_native_blockers_20260910.md) |
| Requirement-by-requirement current Goal closeout | `artifacts/recovery-audit-20260909/goal-completion-audit-002.json` |
| Final focused C++ tests, full CPU regression and evidence refresh | [Validation](h20_final_validation_20260910.md) |

The original-input table includes Python preprocessing/postprocessing, while the
resident and host-model-tensor tables retain separate native Session boundaries.
No historical pilot, RTX/H100 result, incomplete campaign or rejected export is
promoted into the new H20 formal results. Completed GPU campaigns are not rerun.

## Current J6M SmolVLA Delivery

- [H20 and J6M experiment summary](edgefm_h20_j6m_experiment_summary_20260914.md):
  combined platform tables, timing boundaries, output checks and remaining
  Orin/J6M scope.
- [Existing-HBM performance](smolvla_j6m_performance_20260914.md): audited
  five-process CDFs for V2/V5 Prefix, Solver Step-060 and Finish output crop;
  the formal Prefix mean ratio is 2.533x (V2 1402.998 ms, V5 553.930 ms),
  Step mean is 89.921 ms, and Finish crop mean is 0.261 ms. Quantization
  searches and resident full-chain timing remain outside stage delivery.
- [SmolVLA stage-060](smolvla_j6m_stage060_20260911.md): real `j6m-2`
  stage execution, 98.73x same-board speedup, five-process latency CDF,
  complete-step output cosine check, and operator before/after ablation.
- [J6M feasibility](j6m_feasibility_assessment_20260911.md): toolchain,
  operator boundaries, board inventory, and model capacity assessment.
- [Goal status](../edgefm_j6m_goal_status.md): current classification and
  remaining stage measurements and subsequent full-E2E work.

The J6M delivery is stage-level. It does not claim a full SmolVLA prefix plus
ten-step solver pipeline, and it contains no measured RDT, pi0, pi0.5, CogACT or
Qwen board results.

Historical completion gates below do not grant edge-paper acceptance. Qwen native
acceptance and the CogACT same-boundary comparison are the fixed-profile results
indexed above.

## Historical v0.2 Evidence

The following index describes the earlier Host-CUDA paper scope:

- `vlaforge_ir_necessity/`: Invocation IR v0.2 adversarial contract tests;
- `vlaforge_invocation_v02_benchmark/`: Semantic/Plan fixture reference
  benchmark, explicitly not real-model/C++ performance;
- `vlaforge_real_v02/`: real SmolVLA and OpenVLA eager/IR L2 evidence;
- `vlaforge_real_v03/`: real SmolVLA L3/L4, DiffusionDrive L2/L3/L4, and
  OpenVLA-7B L3/L4 Host-CUDA evidence;
- `vlaforge_reproducibility_v01/`: installed-wheel `sm_86` artifact
  evaluation, environment manifest, committed-evidence hashes, reproduction
  commands, and the external large-artifact archive inventory;
- `vlaforge_cuda_matrix_v01/`: paper-grade RTX 3060/CUDA 12.8 matrix with
  two real models, five deterministic workloads, five independent processes,
  eager/direct-AOTI/generated-C++ paths, clustered 95% confidence intervals,
  first-Run/fresh-process/memory data, raw JSON/CSV, and output parity;
- `vlaforge_ablations_v01/`: four formal ablations covering InputRevision
  exact reuse, verified static-arena packing/residency, transaction
  failure/retry, and the direct-AOTI versus generated-C++/clean-wheel
  deployment boundary;
- `vlaforge_autovla_v01/`: held-out real AutoVLA decoder-partition L2 with
  zero new core ops, plus a non-promoted conservative AOTI L3 candidate whose
  final tokens/trajectory pass but intermediate NRMSE exceeds the predeclared
  threshold;
- `vlaforge_minddrive_v01/`: complete MindDrive 0.5B real L3/L4 index for the
  clean five-frame held-out L3 run and clean generated-C++ L4 bundle, covering
  8 logical Regions, 66 physical artifacts, 16 authoritative states, 10 named
  outputs, typed/generic exact parity, revision cache, transaction abort/retry,
  reset, durable archive hashes, and the no-Python boundary;
- `vlaforge_paper_completion_v01/`: mechanical submission audit over the
  required Host-CUDA matrix, ablations, held-out L2, final release gate, paper
  materials and explicit optional/non-blocking evidence;
- `vlaforge_orin_validation.md`: JetPack arm64 VLAForge runtime and generated
  Session portability evidence. Real Orin GPU execution is an optional
  cross-platform extension, not a Host-CUDA paper completion condition.

Current architecture and evidence status:

- `doc/vlaforge_invocation_ir_v0_2.md`;
- `doc/vlaforge_cpp_aot_progress.md`;
- `doc/model_cards/README.md`.

Git history remains the source of provenance for prior experiments.
