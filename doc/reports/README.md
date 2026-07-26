# VLAForge Report Status

Current v0.2 evidence:

- `vlaforge_ir_necessity/`: Invocation IR v0.2 adversarial contract tests;
- `vlaforge_invocation_v02_benchmark/`: Semantic/Plan fixture reference
  benchmark, explicitly not real-model/C++ performance;
- `vlaforge_real_v02/`: real SmolVLA and OpenVLA eager/IR L2 evidence;
- `vlaforge_real_v03/`: real SmolVLA L3/L4, DiffusionDrive L2/L3/L4, and
  OpenVLA-7B L3 Host-CUDA evidence;
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
