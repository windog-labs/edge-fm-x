# DiffusionDrive Model Adaptation Card

| 项 | 当前记录 |
|---|---|
| Upstream | `hustvl/DiffusionDrive@9b52ed0ec06b073d82d6f392ab084c7b301c8681` |
| License / checkpoint | code MIT；checkpoint 按 Hugging Face model card 为 non-commercial |
| Checkpoint | `hustvl/DiffusionDrive@8e3cc29cfdb5aa1a4c0818012f9a250d5153bc71`，SHA256 `008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b` |
| Source entry | `transfuser_agent.py`、`V2TransfuserModel`、`TrajectoryHead.forward_test` |
| 当前证据 | L0 + L1 + real L2 + real Host-CUDA L3 + deterministic fixture-L4 |
| Real Adapter | `diffusiondrive_real.py`，1,241 LOC，复用 `DiffusionPlanner` template |
| Core op 增量 | 0 |

源码审计确认 feature builder 生成 camera/lidar/status tensors；推理 head 使用
20 个 trajectory anchors/candidates、两步 truncated DDIM，并产生 candidate
regression/classification 后选出 best trajectory。真实 Adapter 接收已经准备好的
`[1,3,256,1024]` camera、`[1,1,256,256]` lidar BEV、`[1,8]`
status 和显式 `[1,20,8,2]` noise；传感器构造与同步不进入 IR。

真实 checkpoint 被划分为五个 Region：

- `condition_encoder`：camera/lidar/status exact cache；
- `initialize_planner_state`：显式 noise 与 anchor 生成 loop state；
- `make_denoise_timestep`：有界循环 index 到 10/0 timestep；
- `denoise_planner_step`：loop-carried `[1,820]` SSA；
- `decode_planner_outputs`：20 条候选、scores 和 selected trajectory。

固定 checkpoint 严格加载无 missing/unexpected key。官方
`V2TransfuserModel.forward` 与 Region chain 的 candidate trajectories、
candidate scores、selected trajectory、BEV semantic、agent states 和
agent labels 全部 bit-exact；五个 strict `torch.export` graph 的 replay
误差也全部为 0，effect audit 全部通过。RTX 3060 上该 capture 的峰值 CUDA
allocated 为 307,197,952 bytes。证据见
[`diffusiondrive_frontend_l2.json`](../reports/vlaforge_real_v03/diffusiondrive_frontend_l2.json)。

五个 Region 随后编译为 `sm_86` AOTInductor packages，总大小
248,879,397 bytes、总编译时间 48.70 s。saved exported programs 对 eager
保持 exact；artifact 的全部 Region/output NRMSE 不超过 `1e-3`，最终
trajectory 最大绝对误差 `7.84e-4`、均值 `1.15e-4`、NRMSE
`1.97e-4`，重复 artifact pipeline bit-exact。该结果是数值 L3，不声称
artifact 与 eager bit-exact。完整证据见
[`diffusiondrive_artifact_l3.json`](../reports/vlaforge_real_v03/diffusiondrive_artifact_l3.json)。

VLAForge fixture 仍使用缩小的 K=3、两步 denoise 来验证 generated C++
transaction/cache 行为：
生成的无 Python C++ Session 已逐元素对齐全部 3×6×2 candidates、3 个 scores
和 selected trajectory，并对齐 Semantic/Plan/C++ trace；这仍只是
DiffusionDrive-like fixture，不是 checkpoint 证据。

传感器 stitching/点云 histogram 属于底软或外部 preprocessing Region，不进入
core IR。真实 generated C++ Session、paper-grade latency 和长稳仍待 L4；
fixture-L4 不得标为 real deployment。
