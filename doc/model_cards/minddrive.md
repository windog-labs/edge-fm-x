# MindDrive 0.5B Model Adaptation Card

| 项 | 当前记录 |
|---|---|
| Upstream | `xiaomi-mlab/MindDrive@1a4085dab1c20895a0c8d2b67b4f8e65712fa8de` |
| Code license | Apache-2.0 |
| Dataset boundary | Bench2Drive data is CC BY-NC-ND 4.0 |
| Checkpoint repository | `poleyzdk/Minddrive@5cf1eafc7f6d1028006f2d97d083d8e9aa4c0b12` |
| 当前证据 | real-checkpoint upstream eager + EVA、Qwen decision/action、trajectory decoder strict capture；完整模型仍为 L2-prerequisite-only |
| Adapter | offline eager、13-input Semantic IR/Plan contract、4 个 real TensorRegion 和 held-out execution 已实现；object/map/stateful path 进行中 |
| Core op 增量 | 0（Semantic IR/Plan 与 4 个 real Region 已审计） |

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

已固定连续两个真实 Bench2Drive 六相机帧和对应
measurement/annotation：

- dataset `Telkwevr/Bench2Drive-Speed-sample@c84ffbc2b7f3bda9b4acb7fab6971b3599092f42`；
- route `Accident+Follow_Town12_Road392_Route112425_Weather14_01-07-09-36-50`；
- frame `00400` 作为 backend calibration，frame `00401` 作为锁定门槛后的
  held-out validation；
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
`eager_fp32.json` 明确标记为 `L2-prerequisite-only`。后续显式捕获的两份
Gaussian noise 作为 `trajectory_noise`、`path_noise` InputPort 固定在
`real_invocation_inputs.pt` 中；它们是模型调用输入，不是 Session 内部 RNG。
这样同一调用在 backend failure 后可确定性重试，也不会把近似复用伪装成
exact cache。

## 首个真实 TensorRegion

`vision_encoder` 保留严格加载的 24 层 EVA-ViT 和全部 checkpoint
parameters。上游 evaluation GridMask 是恒等映射但会先执行 Python RNG；
Adapter 将其静态消除。上游 FlashAttention 2 直接调用无 dispatcher schema
的 PyCapsule，无法进入 `torch.export`；Adapter 因此使用同 FP16 Q/K/V、
scale、dropout=0、non-causal 语义的 ATen SDPA backend。该替换不涉及旧
EdgeFM kernel。

数值门槛在 frame `00401` 运行前锁定为：feature max-abs `<=0.5`，以官方
feature absolute maximum 归一化的 RMSE `<=1e-3`。结果为：

| sample | max abs | NRMSE | 结果 |
|---|---:|---:|---|
| `00400` calibration | 0.48854065 | 8.8676e-6 | 通过 |
| `00401` held-out | 0.46158600 | 8.3323e-6 | 通过 |

frame `00400` strict `torch.export` effect audit 通过，capture eager/export
max-abs 为 0；同一 exported program 在 frame `00401` 直接执行并通过不变
门槛。持久证据位于 `capture/vision`：

- graph digest
  `a39d9e3ced87f87e4e87a61b2e0042199286442d520cedc5f51743c5cc816ccd`；
- `vision_encoder.pt2e` SHA256
  `68b453e0d03435c8f8f062644991e56dfff2d713e3091474b44a6449f15cb5df`；
- static ABI `[1,6,3,640,640] f32 -> [1,6,1024,40,40] f32`；
- 24 个 shared rotary-buffer registration aliases 被复制为等值常量，
  仅规范化 module ownership，不改变计算。

## Qwen decision/action TensorRegions

真实 Qwen2-0.5B backbone、官方 PEFT LoRA expert 和真实 object/map head
产生的 529 个 vision tokens 被保留。两个 bounded prefill Region 均 strict
export、effect audit 和 held-out exported execution 通过：

| Region | static ABI | calibration max abs / NRMSE | held-out max abs / NRMSE | artifact SHA256 |
|---|---|---:|---:|---|
| `decision_expert` | `[53] i64 + [1,529,896] f32 -> [1,7] f32` | `1.1921e-5 / 5.8715e-7` | `4.2915e-6 / 2.5535e-7` | `20425d292a630464f774e76cbacd0c6d9ac11119c203f76cc0d9fde730670fc0` |
| `action_expert` | `[71] i64 + [1,529,896] f32 -> [2,896] f32` | `0 / 0` | `0 / 0` | `675cc94a0a704985820be8f3cb7078facd41eefac735aee559877bbc39c9dfce` |

decision Region 只投影上游实际读取的 7 个词表行，是对完整
151k-vocabulary projection 的 exact DCE，不是缩小或替换模型。静态 FP32
RoPE 以等价 ATen 表达移除了 PyTorch 2.4 无法验证的 autocast context。
两个约 2.00 GB 的 exported program 位于 `capture/language`。

## 真实 trajectory decoder TensorRegion

`trajectory_decoder` 保留严格加载的两个 probabilistic distribution、
4-layer GRU predictor、7/6 mode MLP heads、6 步 ego trajectory 和 20 步
path decode。上游两个 `randn_like` 被提升为固定 shape 的显式输入。为绕开
`nn.GRU` 在 strict export 中读取 storage pointer 的 eager-only
flat-weight cache，Adapter 以相同 named parameters 直接调用相同 ATen GRU；
模型结构和权重未裁剪。

门槛在 held-out 前锁定为 max-abs `<=3e-6`、NRMSE `<=1e-6`：

| sample | trajectory max abs | path max abs | speed/path command | 结果 |
|---|---:|---:|---|---|
| `00400` calibration | `9.3132e-10` | `1.9073e-6` | exact / exact | 通过 |
| `00401` held-out | `1.8626e-9` | `9.5367e-7` | exact / exact | 通过 |

strict-export eager parity 为 0，effect audit 无 hidden RNG、mutation 或
external I/O。116,408,192-byte artifact SHA256 为
`b97923e1fbfa01de5928cc004e54b773c62441d00e4fa5d6f3b331a3868debd6`，
证据位于 `capture/trajectory`。

这些结果证明 camera frontend 之后的真实语言与 trajectory partitions 已
可捕获，但尚不等于完整 MindDrive L2：object/map heads、16 个 authoritative
state 的显式 first/stateful path，以及完整 Semantic IR/Plan 仍需闭合。

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

## Region 状态

- `vision_encoder`：六相机 EVA-ViT，real strict capture + held-out
  exported execution 已通过；
- `object_map_encoder`：position/object/map tokens，待显式状态化 capture；
- `decision_expert`：真实 Qwen prefill/meta-action，strict capture +
  held-out 已通过；
- `action_expert`：真实 Qwen trajectory feature，strict capture +
  held-out 已通过；
- `trajectory_decoder`：显式 noise、6 点 trajectory、20 点 path 和两个
  command，strict capture + held-out 已通过；
- `detection_decoder`：scores/labels/motion/boxes，待 capture。

这些边界均使用现有 TensorRegion、structured bounded control 和 named
transactional outputs；不新增 core opcode。

## 当前缺口

- real L2 Semantic IR/Plan parity；
- object/map 和 detection decode 的真实 TensorRegion capture；
- 用上述真实 Region 替换当前 first/stateful monolithic planner
  placeholders，闭合 camera 到全部 named outputs 的数据流；
- 连续两次 Run 的显式 persistent state、ResetEpisode、revision cache
  和 failure/abort 语义；
- real L3 AOTI artifacts；
- real L4 no-Python C++ Session；
- 实测 memory/performance。

在以上证据形成前，本卡片不得升级为 real L2。
