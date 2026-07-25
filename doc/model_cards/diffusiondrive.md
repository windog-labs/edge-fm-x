# DiffusionDrive Model Adaptation Card

| 项 | 当前记录 |
|---|---|
| Upstream | `hustvl/DiffusionDrive@9b52ed0ec06b073d82d6f392ab084c7b301c8681` |
| License / checkpoint | MIT；release checkpoint 未下载 |
| Source entry | `transfuser_agent.py`、`V2TransfuserModel`、`TrajectoryHead.forward_test` |
| 当前证据 | 实际 source audit L0 + L1 + deterministic fixture-L4 |
| Adapter | `build_driving_diffusion_fixture`，221 LOC |
| Core op 增量 | 0 |

源码审计确认 feature builder 生成 camera/lidar/status tensors；推理 head 使用
20 个 trajectory anchors/candidates、两步 truncated DDIM，并产生 candidate
regression/classification 后选出 best trajectory。VLAForge fixture 使用缩小的
K=3、两步 denoise、candidate trajectories + scores + selected trajectory，
验证 loop-carried SSA、exact condition cache 和多 named outputs。
生成的无 Python C++ Session 已逐元素对齐全部 3×6×2 candidates、3 个 scores
和 selected trajectory，并对齐 Semantic/Plan/C++ trace；这仍只是
DiffusionDrive-like fixture，不是 checkpoint 证据。

传感器 stitching/点云 histogram 属于底软或外部 preprocessing Region，不进入
core IR。未完成真实 checkpoint L2/L3/L4；fixture-L4 不得标为 real deployment。
Memory/performance：static arena 已覆盖，真实模型 pending。
