# VLAForge Report Status

Current v0.2 evidence:

- `vlaforge_ir_necessity/`: Invocation IR v0.2 adversarial contract tests;
- `vlaforge_invocation_v02_benchmark/`: Semantic/Plan fixture reference
  benchmark, explicitly not real-model/C++ performance;
- `vlaforge_real_v02/`: real SmolVLA and OpenVLA eager/IR L2 evidence;
- `vlaforge_orin_validation.md`: environment status only; real Orin execution
  remains pending.

All other reports in this directory were generated for the superseded v0.1
tick/epoch architecture. They are retained for provenance and may contain
useful raw model/frontend measurements, but they are not valid v0.2 release or
paper evidence. In particular, `vlaforge_paper_benchmark.{json,csv,md}` and
the real generated-C++ reports must be reproduced through the passive
`Session::Run()` ABI before being cited as current results.

Current architecture and evidence status:

- `doc/vlaforge_invocation_ir_v0_2.md`;
- `doc/vlaforge_cpp_aot_progress.md`;
- `doc/model_cards/README.md`.
