# VLAForge Model Adaptation Cards

每张卡片只陈述当前仓库已经具备的证据。`fixture-L4` 表示 deterministic
fixture 已经生成并运行无 Python C++ Session，不表示真实 checkpoint 达到 L4。

| 模型 | 当前证据 | Core op 增量 | 卡片 |
|---|---:|---:|---|
| RT-1 | L0 + L1 | 0 | [rt1.md](rt1.md) |
| ACT | L0 + L1 | 0 | [act.md](act.md) |
| Octo | L0 + L1 | 0 | [octo.md](octo.md) |
| OpenVLA | L0 + L1 + real L2 + real L3 + fixture-L4 | 0 | [openvla.md](openvla.md) |
| π0 | L0 + L1 | 0 | [pi0.md](pi0.md) |
| SmolVLA | L0 + L1 + real L2 + real L3 + real L4 | 0 | [smolvla.md](smolvla.md) |
| GR00T N1.7 | L0 + L1 | 0 | [groot_n1.md](groot_n1.md) |
| DiffusionDrive | L0 + L1 + real L2 + real L3 + real L4 | 0 | [diffusiondrive.md](diffusiondrive.md) |
| AutoVLA | L0 + L1 + real L2 partition | 0 | [autovla.md](autovla.md) |
| ReCogDrive | L0 + structural L1 | 0 | [recogdrive.md](recogdrive.md) |
| UniDriveVLA | L0 + structural L1 | 0 | [unidrivevla.md](unidrivevla.md) |
| OpenDriveVLA | L0 + structural L1 | 0 | [opendrivevla.md](opendrivevla.md) |

共同验收路径：

- fixture：`vlaforge/tests/models/test_model_fixtures.py`
- upstream pin：`vlaforge/python/vlaforge/adapters/model_contracts.py`
- frozen-core held-out：
  `doc/reports/vlaforge_heldout_v01/heldout_audit.md`
- 生成 C++：`vlaforge/tests/codegen/test_codegen.py`
- 真实模型环境门控：`vlaforge/tests/models/test_real_*.py`

卡片中的 memory/performance 未实测项必须保持 `pending`，不得用静态 shape
估算冒充真实峰值显存或延迟。
