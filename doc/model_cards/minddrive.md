# MindDrive 0.5B Model Adaptation Card

| 项 | 当前记录 |
|---|---|
| Upstream | `xiaomi-mlab/MindDrive@1a4085dab1c20895a0c8d2b67b4f8e65712fa8de` |
| Code license | Apache-2.0 |
| Dataset boundary | Bench2Drive data is CC BY-NC-ND 4.0 |
| Checkpoint repository | `poleyzdk/Minddrive@5cf1eafc7f6d1028006f2d97d083d8e9aa4c0b12` |
| 当前证据 | **完整 real L4**：real L2/L3 保持；8 logical Regions 由 66 个真实 `sm_86` AOTI artifacts 执行，生成的 no-Python C++ Session 在五帧序列上通过 typed/generic API、事务、cache、reset 与 10 named outputs exact parity |
| Adapter | 13 个静态 InputPort、8 个 TensorRegion、16 个 StateSlot、1 个 transactional output group；无 sensor/timer/middleware 语义 |
| Core op 增量 | **0** |
| 当前缺口 | generated C++ 的正式 cold/first/warm、RSS/CUDA memory、revision 矩阵与 1000-Run soak 已完成；尚未形成与 SmolVLA/DiffusionDrive 相同协议的 eager/direct-AOTI/generated-C++ 三路径性能对照，不影响 real L4 correctness 等级 |

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
的同一路线，连续使用 frame `00400` 至 `00404`。`00400` 至 `00403`
用于 L3 development/calibration；`00404` 是在 numerical contract v3
冻结后才执行的 compiled held-out。六相机原图、
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
| `position_encoder` | 41,236,298 | `daf7271a3b924f35b719bd7660a68998faa30b17dc07bba4eb5c400bd91e94bf` |
| `map_encoder` | 96,807,824 | `fd414c2d7e0ffd0d85e612e59a57a924ab3e04064ec1db4c1adaf570c40a1bd7` |
| `detection_encoder` | 116,519,868 | `5ed7e9ad2a7a9cb11440d21dba4d4acfa18c2783614ca6d08d8dd7f663fbc0a9` |
| `decision_expert` | exported program | `4ba68bc01c7caf5574e89a2d8df6b843e1417f171d6ea9b0d467524b7ae06315` |
| `action_expert` | exported program | `55eb2084c467a5bd464d906f5620f2c1f3d523902b2aa5408be30f598db5296b` |
| `trajectory_decoder` | 116,408,192 | `b97923e1fbfa01de5928cc004e54b773c62441d00e4fa5d6f3b331a3868debd6` |
| `detection_decoder` | 838,986 | `c3599c2f02753ef2bc7632857f8ca0f45b4a13c03b7011c3ad9665cd41f96e6f` |

上游 FlashAttention 2 通过无 dispatcher schema 的 PyCapsule 调用，无法直接
进入当前 `torch.export`。曾捕获的 ATen SDPA vision artifact 在 frame
`00400/00401` 的局部门槛内通过，但在预先锁定的完整 pipeline held-out
门槛上分别暴露 vision/detection-token 和低置信检测排序偏差。两个失败报告
被原样保留，门槛没有事后放宽：

- `capture/pipeline/heldout_00400_00401_00402.json`；
- `capture/pipeline/heldout_v2_00400_00401_00402_00403.json`。

完整 L2 因此使用 source-exact FlashAttention vision 作为静态 Tensor
Region plugin；它仍只接受/返回静态 Tensor ABI，不把任意 Python/CARLA
宿主对象泄露进 IR。L3 将 logical vision Region 物理分解为 stem、
24 组 pre/FlashAttention/post 和 finish：50 个 ATen 部分由 AOTI 编译，
中间调用上游已编译 FlashAttention CUDA extension。该分解不增加 core op。

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

## 真实 held-out 端到端 L3

第一轮 compiled `00400 -> 00403` 暴露了两个 L2 无法覆盖的边界：EVA
physical decomposition 的 heavy-tail feature maximum，以及 proposal state
经过四次 compiled carry 后不能复用单 Region 阈值。该序列被明确降级为
development data。contract v3 在获取 `00404` 后、执行其 compiled pipeline
前冻结：所有 task-output 阈值保持不变；vision NRMSE 仍为 `2e-5`；state
增加单独的数值门槛和 geometry-assignment 最大距离门槛。

新的 `00400 -> 00404` held-out 使用 64 个真实 `sm_86` AOTI Regions：

- vision：50 个 physical Regions + upstream compiled FlashAttention；
- map：front + 6 decoder layers + finish；
- position、detection encoder/decoder、decision/action experts 和 trajectory
  decoder：6 个 Regions。

结果全部通过：

| 关键证据 | held-out 结果 |
|---|---:|
| vision feature | max `0.83009`，NRMSE `1.1055e-5` |
| detection tokens | max `1.5151e-3`，NRMSE `2.8654e-5` |
| decision/action | max `5.1403e-4` / `2.8334e-3` |
| trajectory/path | max `6.4123e-4` / `6.0374e-4` |
| speed/path command | exact |
| detection set | count exact，center p95 `3.3221e-3 m` |
| detection state assignment | maximum `1.6850e-2 m` |
| map state assignment | maximum `1.5171e-1 m` |
| valid mask/count | exact |

统一 artifact manifest 精确覆盖 64 个执行对象，拒绝未使用的旧 logical
`map_encoder.so`，并逐 Region 绑定 capture SHA256、compile report、
contiguous physical ABI、真实 `.so` SHA256 和非硬链接不可变性。
`artifact_set_sha256` 为
`772aef7b066aff8a2068cb0cc78742ad0f486ebb21dd3df5171c7872ea71192b`。
正式报告：

- `artifacts/reports/aoti24_all_contiguous_v6/artifact_manifest.json`；
- `artifacts/reports/heldout_l3_aoti24_v6_contract_v3_00400_00401_00402_00403_00404.json`。

因此 MindDrive 已达到完整 `real-L3-compiled-held-out-end-to-end`；它不是
compile-only、部分 decoder 或 fixture 证据。

## 真实 generated no-Python C++ L4

L4 保持 8 个逻辑 TensorRegion，不向 Semantic IR 增加模型专属 op。物理
backend 将 vision/map 两个逻辑 Region 表达为静态
`vlaforge.aoti_sequence/1`：

- vision sequence：52 个 artifact、74 个 node、123 个静态 value；
- map sequence：8 个 artifact、8 个 node、39 个静态 value；
- 其余 position/detection/decision/action/trajectory/detection-decode：
  6 个 direct raw AOTI `.so`；
- 合计 66 个真实 physical artifacts，全部由 bundle SHA256 清单覆盖。

Sequence 只是一种受验证的 TensorRegion backend artifact，不是新的 core
IR 控制语义。边值采用 canonical dense ABI；若 AOTI 返回 padded/strided
view，provider 在物理边界显式物化 contiguous tensor。临时值按最后一次
使用释放，避免 24 层 vision 中间量全部常驻。16 个大状态通过 runtime
zero initializer 建立，不在生成源码中嵌入巨大字面量。

clean-worktree bundle 从 commit
`b9372682abd9af484a046014be6305dd2ae1ee45` 构建，记录
`source_dirty=false`：

| 发布物 | SHA256 |
|---|---|
| `l4/bundle/bundle.json` | `fa7a2ef0109c6598a31166f29ce1406e5557ca3b730188fe3d51310f93c78c5b` |
| `l4/bundle/bin/vlaforge_generated_runner` | `b48174f1c1def6c02df2c92180fe51ae07040838e9955d8e5b079145ea10390b` |
| `l4/evidence/minddrive-real-l4.json` | `20cef99bef75ee60a5d43bd62b731d25451640b1f694be9dfe3c339eba885a6e` |

runner 使用 LibTorch 2.4.1+cu118，在 RTX 3060 `sm_86` 上运行；传入无效
`PYTHONHOME/PYTHONPATH` 仍成功，`ldd` 无 `libpython`。模型专属强类型
wrapper 与通用 C ABI 的 10 个最终输出逐 bit 相等，并与真实 compiled
reference 逐 bit 相等。

typed trace 覆盖一个 same-revision cache hit、8 个 cache miss、8 个成功
事务、1 个 validation abort、1 次 episode reset，以及 16 states × 8
成功 Run = 128 次 state commit：

```text
TRACE_SUMMARY,1,8,128,8,1,8,1
```

NaN trajectory failure 不推进状态或替换 output group；随后新 revision
retry 正常提交。L4 参考采用已通过 frozen contract-v3 的真实 SDPA/AOTI
compiled path，作用是验证部署等价与事务边界；它不被描述成新增的模型
泛化样本。

## 资源与性能记录

官方完整 eager 的已记录峰值：

- CUDA allocated：4,123,524,096 bytes；
- CUDA reserved：5,442,109,440 bytes。

在 IR/Plan L2 验证中，source-exact vision、16-state Session 和逐调用释放的
export providers 可在 RTX 3060 12GB 上连续执行；修正 provider 的
inference-only/no-grad 调用契约后，单次调试峰值约 5.68 GB。该数值是开发
诊断，不替代下面的 fresh-process generated C++ 正式统计。

generated no-Python C++ Session 使用四种 revision 模式，各执行 5 个独立
fresh process、每进程 1 次 warmup 加 10 次测量，并以进程为 bootstrap
单元执行 2,000 次重采样：

| Revision mode | Init mean | First Run mean | Warm mean | 95% CI of warm mean |
|---|---:|---:|---:|---:|
| full | 4083.01 ms | 1388.93 ms | 1270.38 ms | [1263.47, 1276.01] ms |
| same | 4082.14 ms | 1392.75 ms | 260.01 ms | [259.86, 260.16] ms |
| new | 4075.96 ms | 1395.93 ms | 1279.27 ms | [1277.59, 1280.66] ms |
| missing | 4073.88 ms | 1398.62 ms | 1281.75 ms | [1281.26, 1282.16] ms |

每个 same-revision 进程均记录 10 hit/0 miss；full/new/missing 均记录
0 hit/10 miss。same 相对 new 的端到端加速为约 4.92×。四类声明内存为：

- external inputs：29,493,452 bytes/Run，13 个输入；
- external outputs：29,332 bytes/Run，10 个输出；
- per-Run static arena：56,559,808 bytes；
- authoritative state arena：3,351,680 bytes，16 个 states；
- derived cache：39,321,600 bytes，1 个物理 buffer。

额外的 1000-Run same-revision soak 记录 1000 cache hits、0 misses、16,000
state commits 和 1000 transaction/output commits。16 个 state version
均为 1001（1 次 warmup + 1000 次正式 Run），CUDA memory drift 为 0，
Host RSS drift 为 +60 KiB。正式索引为：

- `doc/reports/vlaforge_minddrive_v01/minddrive_l4_benchmark.json`；
- `doc/reports/vlaforge_minddrive_v01/minddrive_l4_benchmark.md`；
- `doc/reports/vlaforge_minddrive_v01/minddrive_l4_soak.json`。

## 后续可选证据

1. 使用相同真实五帧、同一批 artifacts 和常驻 provider 补充
   eager/direct-AOTI/generated-C++ 三路径对照；在完成前不能把上述
   generated-only 数字解释为 direct-artifact orchestration overhead；
2. 按模型适用性补充 Region 粒度、derived-cache 或 validation overhead
   消融，但不改变当前 L4 correctness 结论；
3. 在第二台 GPU 或 Orin 上复现 artifact；它们属于跨硬件增强，不是当前
   Host-CUDA L4 的完成条件。

论文现在可以声称 VLAForge 已在 RTX 3060 上将真实、持久状态化的自动驾驶
VLA 编译为 verified no-Python C++ Session；不能据此声称跨 GPU、Orin
性能、实时闭环或功耗。
