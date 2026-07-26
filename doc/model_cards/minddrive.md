# MindDrive 0.5B Model Adaptation Card

| 项 | 当前记录 |
|---|---|
| Upstream | `xiaomi-mlab/MindDrive@1a4085dab1c20895a0c8d2b67b4f8e65712fa8de` |
| Code license | Apache-2.0 |
| Dataset boundary | Bench2Drive data is CC BY-NC-ND 4.0 |
| Checkpoint repository | `poleyzdk/Minddrive@5cf1eafc7f6d1028006f2d97d083d8e9aa4c0b12` |
| 当前证据 | real-checkpoint upstream eager reference（L2-prerequisite-only）；尚未声明 L2 |
| Adapter | offline eager probe 已实现；Semantic IR real adapter 进行中 |
| Core op 增量 | 0（目标；真实 capture 后重新审计） |

## 固定发布物

完整 0.5B 离线推理集合共 9,808,889,451 bytes：

| 文件 | bytes | SHA256 |
|---|---:|---|
| `minddrive_rltrain.pth` | 6,593,355,869 | `39c86eddeaf57b15b9aeb54beb9f698539f6ee1e83529724b1bfc8c5e11b4ba0` |
| `llava-qwen2-0.5b/model.safetensors` | 1,892,090,688 | `6fc9882475867279ee66e505ded47b5d722fc09b0d34bc7684a26080d662825f` |
| `llava-qwen2-0.5b-eva02_petr_proj.pth` | 1,307,562,253 | `1feabeea917d46678514eb9160a2108733569608126daa2eb481431c6f94d38e` |

其余 15.88 MiB 是固定 revision 下的 tokenizer/config 文件。全部 11 个
文件已下载到 durable archive 并完成本地全文件 SHA256 复核。

## 真实输入与模型边界

官方部署 profile 是 batch 1、六路 1600×900 RGB 相机，经官方 pipeline
resize/normalize 到六路 640×640。外部还提供 camera calibration、ego pose、
speed/can-bus、route command 和语言 prompt。VLAForge 只接收这些准备好的
Tensor/Scalar 输入，不接管 CARLA、时间同步、route planner 或 PID 控制。

已固定一个真实 Bench2Drive 六相机帧和对应 measurement/annotation：

- dataset `Telkwevr/Bench2Drive-Speed-sample@c84ffbc2b7f3bda9b4acb7fab6971b3599092f42`；
- route `Accident+Follow_Town12_Road392_Route112425_Weather14_01-07-09-36-50`；
- frame `00400`；
- 六张 RGB 图均为 1600×900；记录包含 speed、command、ego position、
  heading、acceleration 和 angular velocity；
- 文件位于
  `/home/zhangzimo/Archives/vlaforge-minddrive-0.5b-20260726/real_input`。

该数据只用于离线输入重建与数值 parity，不重新分发进 Git。

## 已通过的真实 eager 基线

在隔离的 Python 3.10、PyTorch 2.4.1+cu118、flash-attn 2.6.3 与
本地编译 `sm_86` MMCV CUDA extension 环境中，已完成：

- 官方 config 和 `build_model` 路径；
- checkpoint 2,431 keys 到 1,895 个推理态 keys 的精确投影，
  `strict=True` 无 missing key，536 个额外 key 均属于已审计的 RL
  critic 或 non-persistent rotary buffer；
- 官方六相机/VQA preprocessing 和完整
  `Minddrive.forward_test/simple_test`；
- 八个 named outputs 和 16 个跨 Run persistent-state tensors 的导出；
- RTX 3060 FP32 单次 eager forward 成功，峰值 allocated CUDA memory
  4,123,524,096 bytes，峰值 reserved CUDA memory 5,442,109,440 bytes。

耐久化证据位于
`/home/zhangzimo/Archives/vlaforge-minddrive-0.5b-20260726/frontend`。
`eager_fp32.json` 明确标记为 `L2-prerequisite-only`：它尚未证明 capture、
Semantic IR、Plan、事务、cache 或连续 Run parity。

## 上游完整推理链

1. EVA-ViT 编码六路相机；
2. object/map heads 形成 detection 和 lane tokens；
3. Qwen2-0.5B decision expert 产生 speed/path meta-action；
4. action expert 从 vision tokens、prompt 和 meta-action 产生 trajectory
   hidden features；
5. trajectory heads 返回 6 点 ego trajectory、20 点 path，以及检测、
   lane、meta-action 等 named auxiliary outputs。

`team_code/minddrive_b2d_agent.py` 中的 CARLA sensor wrapper、route assembly、
PID 和 `VehicleControl` 属于外层系统，不进入 VLAForge。离线 L2 从官方
`Minddrive.forward_test/simple_test` 开始，保留 camera、prompt、ego/route
到 trajectory 的完整模型路径。

## 计划 Region

- `vision_encoder`：六相机 EVA-ViT；
- `object_map_encoder`：position/object/map tokens；
- `decision_expert`：Qwen prefill/meta-action；
- `action_expert`：trajectory feature；
- `trajectory_decode`：6 点 trajectory、20 点 path 和 aux outputs。

这些边界均使用现有 TensorRegion、structured bounded control 和 named
transactional outputs；不新增 core opcode。

## 当前缺口

- real L2 Semantic IR/Plan parity；
- 连续两次 Run 的显式 persistent state、ResetEpisode、revision cache
  和 failure/abort 语义；
- real L3 AOTI artifacts；
- real L4 no-Python C++ Session；
- 实测 memory/performance。

在以上证据形成前，本卡片不得升级为 real L2。
