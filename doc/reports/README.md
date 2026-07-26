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
- `vlaforge_orin_validation.md`: JetPack arm64 VLAForge runtime and generated
  Session portability evidence. Real Orin GPU execution is an optional
  cross-platform extension, not a Host-CUDA paper completion condition.

Current architecture and evidence status:

- `doc/vlaforge_invocation_ir_v0_2.md`;
- `doc/vlaforge_cpp_aot_progress.md`;
- `doc/model_cards/README.md`.

Git history remains the source of provenance for prior experiments.
