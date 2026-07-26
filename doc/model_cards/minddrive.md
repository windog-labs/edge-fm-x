# MindDrive 0.5B Model Adaptation Card

| 项 | 当前记录 |
|---|---|
| Upstream | `xiaomi-mlab/MindDrive@1a4085dab1c20895a0c8d2b67b4f8e65712fa8de` |
| Code license | Apache-2.0 |
| Dataset boundary | Bench2Drive data is CC BY-NC-ND 4.0 |
| Checkpoint repository | `poleyzdk/Minddrive@5cf1eafc7f6d1028006f2d97d083d8e9aa4c0b12` |
| 当前证据 | **完整 real L2**：真实六相机 frontend、完整 checkpoint、8 个显式 Region、16 个 authoritative StateSlot、10 个 named outputs，eager/captured/Semantic IR/Plan 连续四帧 parity 通过 |
| Adapter | 13 个静态 InputPort、8 个 TensorRegion、16 个 StateSlot、1 个 transactional output group；无 sensor/timer/middleware 语义 |
| Core op 增量 | **0** |
| 当前缺口 | real L3 AOTI、real L4 no-Python C++、正式性能/内存矩阵 |

## 固定发布物

完整 0.5B 离线推理集合共 9,808,889,451 bytes：

| 文件 | bytes | SHA256 |
|---|---:|---|
| `minddrive_rltrain.pth` | 6,593,355,869 | `39c86eddeaf57b15b9aeb54beb9f698539f6ee1e83529724b1bfc8c5e11b4ba0` |
| `llava-qwen2-0.5b/model.safetensors` | 1,892,090,688 | `6fc9882475867279ee66e505ded47b5d722fc09b0d34bc7684a26080d662825f` |
| `llava-qwen2-0.5b-eva02_petr_proj.pth` | 1,307,562,253 | `1feabeea917d46678514eb9160a2108733569608126daa2eb481431c6f94d38e` |

其余 15.88 MiB 是固定 revision 下的 tokenizer/config 文件。全部 11 个
文件位于
`/home/zhangzimo/Archives/vlaforge-minddrive-0.5b-20260726/upstream`，
并完成全文件 SHA256 复核。checkpoint 2,431 keys 被投影为 1,895 个推理态
keys，`strict=True` 无 missing key；536 个额外 key 均属于已审计的 RL
critic 或 non-persistent rotary buffer。

## 真实输入与部署边界

官方 profile 是 batch 1、六路 1600×900 RGB，经官方 pipeline
resize/normalize 为 `[1,6,3,640,640]`。VLAForge 的声明式输入还包括：

- decision/planning token IDs；
- route command、CAN bus、lidar-to-image 和相机内参；
- timestamp、ego pose/inverse pose 和 route-command index；
- 两个显式 planner noise tensors。

后两项 noise 原本来自上游 `randn_like`，现在作为调用输入固定，因此
backend failure 后可以确定性重试。InputRevision 只表示这组输入的数据
身份，不承担相机同步或帧率语义。CARLA、route planner、PID、
`VehicleControl`、传感器同步和动作发布均在 Session 外部。

真实样本固定自
`Telkwevr/Bench2Drive-Speed-sample@c84ffbc2b7f3bda9b4acb7fab6971b3599092f42`
的同一路线，连续使用 frame `00400`、`00401`、`00402` 和 `00403`。`00400`
用于开发校准，后续帧用于连续状态与 held-out 验证；六相机原图、
measurement、annotation、重建后的 13 输入及 reference state/output 均位于
durable archive，不提交到 Git。

## 完整显式 IR

Adapter 没有保留早期的 monolithic first/stateful planner placeholder，而是
构造以下 8 个模型 Region：

1. `vision_encoder`：24 层 EVA-ViT，六相机到 dense features；
2. `position_encoder`：相机几何与位置编码；
3. `map_encoder`：读取并返回 6 个 map authoritative states，同时产生
   map tokens/classes/coordinates；
4. `detection_encoder`：读取并返回 10 个 detection authoritative states，
   同时产生 detection tokens/classes/boxes/motion；
5. `decision_expert`：真实 Qwen2-0.5B + LoRA decision expert；
6. `action_expert`：真实 Qwen2-0.5B + LoRA action expert；
7. `trajectory_decoder`：GRU/MLP trajectory/path decode；
8. `detection_decoder`：固定容量 top-300 scores/labels/boxes/motion 与
   valid mask/count。

16 个 StateSlot 全部采用
`read latest -> stage_write -> transaction commit`。它们是 authoritative
persistent state，不与 derived vision cache 混用。单次成功 Run 后每个
version 恰好 `+1`；validation/backend failure 不推进 version。输出组包含：

- `trajectory`、`path_trajectory`；
- `speed_command`、`path_command`；
- `detection_scores`、`detection_labels`、`detection_boxes`；
- `motion_trajectories`；
- `detection_valid_mask`、`detection_valid_count`。

整个数据流只使用冻结的 Semantic IR v0.2 核心 op，`core_op_delta=0`。

## Region capture

除 source-exact vision plugin 外，其余 7 个 Region 均已 strict
`torch.export`，effect audit 无 hidden RNG、hidden mutation 或 external
I/O：

| Region | artifact bytes | SHA256 |
|---|---:|---|
| `position_encoder` | 41,236,298 | `35d0efcb66cd8a1593325632c9e0e2b722160d5953fef75b753e1e8ce84a23ad` |
| `map_encoder` | 96,807,824 | `a1a0bceb14150e19f4118e4638bc16ae34db70dcbeaa2f320bf93e734ab6b644` |
| `detection_encoder` | 116,517,839 | `c67ef0ef9d6a22767208c402931bc25a96f7d2001c1412e872a868982e3f4915` |
| `decision_expert` | exported program | `4ba68bc01c7caf5574e89a2d8df6b843e1417f171d6ea9b0d467524b7ae06315` |
| `action_expert` | exported program | `55eb2084c467a5bd464d906f5620f2c1f3d523902b2aa5408be30f598db5296b` |
| `trajectory_decoder` | 116,408,192 | `b97923e1fbfa01de5928cc004e54b773c62441d00e4fa5d6f3b331a3868debd6` |
| `detection_decoder` | 838,986 | `bae163c98c3af5988242d044b1001932eb613fe682fe90798428608a4a4bf90b` |

上游 FlashAttention 2 通过无 dispatcher schema 的 PyCapsule 调用，无法直接
进入当前 `torch.export`。曾捕获的 ATen SDPA vision artifact 在 frame
`00400/00401` 的局部门槛内通过，但在预先锁定的完整 pipeline held-out
门槛上分别暴露 vision/detection-token 和低置信检测排序偏差。两个失败报告
被原样保留，门槛没有事后放宽：

- `capture/pipeline/heldout_00400_00401_00402.json`；
- `capture/pipeline/heldout_v2_00400_00401_00402_00403.json`。

完整 L2 因此使用 source-exact FlashAttention vision 作为静态 Tensor
Region plugin；它仍只接受/返回静态 Tensor ABI，不把任意 Python/CARLA
宿主对象泄露进 IR。该 provider 在 source reference 上 bit-exact。后续 L3
必须将它编译为稳定的 no-Python C++/CUDA Region，或明确保留为 L3 blocker。

## 真实 held-out 端到端 L2

预先锁定 numerical contract v2 后，source-exact vision plugin 与 7 个
strict-export Regions 在连续 `00400 -> 00401 -> 00402 -> 00403` 上通过：

| 关键输出 | held-out 误差 |
|---|---:|
| vision feature | bit-exact |
| detection tokens | max `4.4322e-4`, NRMSE `8.2193e-6` |
| decision logits | max `2.4295e-4` |
| action hidden | max `5.4550e-4` |
| trajectory | max `1.0586e-4`, NRMSE `3.4620e-6` |
| path trajectory | max `2.6461e-5` |
| speed/path command | exact |
| detection set | center p95 `4.1199e-4 m`, 100% within `0.15 m` |

报告：
`capture/pipeline/heldout_l2_flash_plugin_00400_00401_00402_00403.json`。

相同 4 帧随后通过真实 Semantic IR 和 Plan executor：

- 4 帧的 10 个 named outputs 全部 exact；
- normalized Semantic/Plan trace exact；
- 16 个 state version 均为 4；
- same revision 的 derived vision cache 命中；
- new revision miss；
- missing revision 两次 Run 均 miss、零 hit；
- validation failure 抛错且 state/output 不变；
- retry 只提交一次；
- `ResetEpisode` 后 state version 全为 0，旧输出不可读取。

报告：
`capture/pipeline/heldout_l2_semantic_ir_plan_00400_00401_00402_00403.json`。
这构成完整 `real-L2`，而不是 fixture、随机权重、decoder partition 或
只编译未执行的 artifact。

## 资源记录

官方完整 eager 的已记录峰值：

- CUDA allocated：4,123,524,096 bytes；
- CUDA reserved：5,442,109,440 bytes。

在 IR/Plan L2 验证中，source-exact vision、16-state Session 和逐调用释放的
export providers 可在 RTX 3060 12GB 上连续执行；修正 provider 的
inference-only/no-grad 调用契约后，单次调试峰值约 5.68 GB。该数值是开发
诊断，不替代后续按论文 workload 采集的 cold/first/warm、RSS、CUDA
allocated/reserved 正式统计。

## 下一证据等级

1. 为 7 个 exported Regions 编译 `sm_86` AOTI artifact；
2. 为 FlashAttention vision 建立稳定 no-Python C++/CUDA Region provider，
   或记录不能满足 source-exact 数值门槛的正式 blocker；
3. 完成 exported/AOTI artifact 与 real L2 reference parity、两次执行、
   artifact hash/size/load/latency/memory 记录，达到 real L3；
4. 若 L3 通过，生成 verified Compile Bundle、typed/generic ABI 和
   no-Python C++ Session，完成 cache/transaction/reset/failure/retry；
5. 将 MindDrive 加入正式性能矩阵、消融、论文表格与 claim-evidence map。

在第 3 步完成前，论文只能声称 MindDrive 达到完整 real L2，不能声称
VLAForge 已编译真实自动驾驶 VLA；在第 4 步完成前不能声称 MindDrive L4。
