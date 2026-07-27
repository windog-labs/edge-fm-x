# VLAForge A100/H20 高显存开发交接

> 状态：Host-CUDA 论文必须项已经完成。本交接只用于补强 AutoVLA 完整
> camera-to-trajectory 证据和跨 GPU 复现，不能改写现有 RTX 3060 实测结论。

## 1. 推荐机器

优先使用 H20 或 A100 80GB；A100 40GB 也可先做 sequential Region capture，
但更可能需要 CPU offload 和逐 Region 编译。代码不按 GPU 名称判断，而是自动
读取 CUDA capability、显存和驱动：

- A100 通常生成 `sm_80` artifact；
- H20 通常生成 `sm_90` artifact；
- 原 RTX 3060 的 `sm_86` AOTI package 不能跨机器复用。

`run_high_memory_autovla.py` 会把目标写入 compile report，并在加载前拒绝
target mismatch。每个阶段独立启动 Python 进程，避免 frontend、编译器和
artifact loader 的内存累积。

## 2. 需要传到目标机器的内容

代码和小型配置：

1. 当前 `codex/vlaforge-paper-artifact` 分支或交接 Git bundle；
2. `AutoVLA@ba34eed74ce6729e7986592d0e66cbaca397b4fa` 源码；
3. `codebook_cache/agent_vocab.pkl`；
4. `config/eval/qwen2.5-vl-3B-nusc-sft-eval.yaml`，仅供完整官方入口使用；
5. 一份真实、离线、已经由外部系统组装好的三相机四帧输入。

大型文件：

1. `AutoVLA_PDMS_89.ckpt`，16,292,664,780 bytes，SHA256
   `58246773393da45678a3f35d354fd969eed6833ecc8ee596edc5e283d1a87473`；
2. `Qwen/Qwen2.5-VL-3B-Instruct` revision
   `66285546d2b821cf421d4f5eb2576359d3770cd3` 的本地 snapshot；
3. 可选：当前 partition L2 exports，用于快速对照，但 AOTI package 必须在
   目标机器重新编译。

本机 checkpoint 的耐久路径为：

```text
/home/zhangzimo/Archives/vlaforge-paper-artifact-20260726/required/vlaforge-autovla-checkpoint-a7d7ba3/AutoVLA_PDMS_89.ckpt
```

不要把 checkpoint、Qwen 权重、`.pt2` 或输入图像提交到 Git。

## 3. 环境

建议使用 Python 3.11 和独立环境。PyTorch、CUDA runtime 与目标驱动必须
匹配；为了与当前编译链对齐，优先安装与本机一致的 PyTorch
`2.10.0+cu128`，再安装：

```bash
python -m pip install -e '/path/to/edge-fm-x/vlaforge[test]'
python -m pip install \
  -r /path/to/edge-fm-x/vlaforge/requirements/high-memory-autovla.txt
```

这里只安装离线模型部署需要的依赖，不安装 nuPlan、Ray、训练或评测栈。
VLAForge 不读取数据集、不做传感器同步，也不要求 CARLA/ROS/Cyber。
AutoVLA 上游模块在文件顶层导入 nuPlan score，即使 `AutoVLA.predict` 完全不
使用它；full-eager probe 只为这些训练评分符号安装 fail-closed import shim，
不会替换或修改 processor、vision/VLM、generation、action tokenizer 或
trajectory rollout。

复制环境模板并修改绝对路径：

```bash
cp \
  /path/to/edge-fm-x/vlaforge/scripts/high_memory/autovla.env.example \
  /path/to/private/autovla.env
set -a
source /path/to/private/autovla.env
set +a
```

`VLAFORGE_AUTOVLA_QWEN_CONFIG` 必须指向 Qwen snapshot 内的
`config.json`；它供 partition frontend 构造 Qwen MLP，不能指向 AutoVLA
源码中的 eval YAML。后者只通过完整 eager probe 的 `--config` 参数传入。

正式证据应在干净 Git revision 上运行。开发中如确需运行，可使用
`--allow-dirty`，但该结果不能直接进入论文。

## 4. 第一阶段：目标机基线

先只做硬件、源码和 checkpoint 审计：

```bash
cd /path/to/edge-fm-x
python vlaforge/tools/run_high_memory_autovla.py \
  --through preflight
```

默认门槛：

- VRAM 至少 39 GiB；
- Host RAM 至少 60 GiB；
- 结果盘空闲至少 80 GiB；
- BF16 GPU；
- AutoVLA、checkpoint、codebook 和 config revision/hash 全部匹配。

如果只想检查 A100 的完整命令而不执行：

```bash
python vlaforge/tools/run_high_memory_autovla.py \
  --target sm_80 \
  --through partition-l3 \
  --print-plan
```

在目标机器重跑现有真实 partition L2→编译→L3：

```bash
python vlaforge/tools/run_high_memory_autovla.py \
  --through partition-l3
```

默认使用 `--precision-mode fp32-internal`：输入/输出仍保持原 BF16/F32
Region ABI，只把 Qwen MLP 与 action projection 的内部 GEMM 提升为 FP32。
export 之前必须先对同一真实权重、同一输入运行 source-BF16 对照，并满足
action token 完全一致、trajectory max-abs 不超过 `2e-3`；报告会保留
hidden/logits 的完整误差，而不会把该变换伪装成 bit-exact。随后
legalized eager→AOTI 仍执行原先的 `1e-3` Region NRMSE 和 `2e-3`
trajectory 门槛。`--precision-mode source-bf16` 仅用于诊断原始 BF16
kernel 数值漂移。

该命令保留预声明 `1e-3` Region NRMSE 和 `2e-3` trajectory max-abs
门槛。A100/H20 上若仍未通过，不得事后放宽；保存
`partition_l3/autovla_artifact_l3.json` 为该硬件上的正式 candidate 结果。
中断后可以加 `--resume`，已完成且 target 一致的阶段才会被跳过。

## 5. 第二阶段：官方完整 eager 入口

复制并填写输入模板：

```text
vlaforge/spec/autovla_full_input.example.json
```

输入必须包含三路相机、每路四帧、车辆速度/加速度、导航指令和可选
`revision`。帧窗口及同步由外部系统准备；probe 只解析已经准备好的文件。

执行：

```bash
python vlaforge/tools/probe_real_autovla_full_eager.py \
  --source-root "$VLAFORGE_AUTOVLA_SOURCE_ROOT" \
  --checkpoint "$VLAFORGE_AUTOVLA_CHECKPOINT" \
  --config "$VLAFORGE_AUTOVLA_SOURCE_ROOT/config/eval/qwen2.5-vl-3B-nusc-sft-eval.yaml" \
  --codebook "$VLAFORGE_AUTOVLA_CODEBOOK" \
  --qwen-model-root "$VLAFORGE_AUTOVLA_QWEN_MODEL_ROOT" \
  --input-manifest /absolute/path/to/real_sample.json \
  --output-dir "$VLAFORGE_HIGH_MEMORY_OUTPUT_ROOT/full_eager" \
  --report "$VLAFORGE_HIGH_MEMORY_OUTPUT_ROOT/full_eager/report.json"
```

该 probe：

- 严格加载官方完整 checkpoint，missing/unexpected key 必须为 0；
- 执行官方 `AutoVLA.predict`；
- 固定 RNG 后用可捕获路径复跑 processor、vision/VLM generation、action
  token extraction 和 trajectory decode；
- 固化真实 prompt tensors、generated IDs、action tokens 与 trajectory；
- 记录 load/generation 时间、Host RSS、CUDA allocated/reserved；
- 只标记为 `L2-candidate-full-real-checkpoint-eager`。

只有后续 eager/captured Regions/Semantic IR/Plan 全部对齐后才能升级为完整
real L2。

## 6. 完整 AutoVLA 的模块落点

不要修改冻结的 15-op core。新增代码应限制在 Adapter、Region、artifact 和
真实模型审计工具：

| 模块 | 职责 |
|---|---|
| `python/vlaforge/adapters/autovla_full.py` | 声明处理后视觉 tensor、prompt/token、ego/route、valid count/mask、revision 及多 named outputs |
| `tools/capture_real_autovla_full.py` | 从 full-eager capture 划分并 strict-export Regions |
| `tools/audit_real_autovla_full.py` | 官方 eager、export、Semantic IR、Plan parity 与 cache/transaction/reset |
| `tools/compile_real_aoti_exports.py` | 在当前 GPU 原生编译各 Region |
| `tools/build_real_autovla_full_l4.py` | verified bundle、typed/generic ABI、no-Python Session |
| `tests/models/test_real_autovla_full*.py` | opt-in L2/L3/L4 gate |

推荐 Region：

1. `autovla_vision_encoder_projector`；
2. `autovla_vlm_prefill`；
3. `autovla_decode_step`；
4. `autovla_action_detokenize`。

约束：

- 三相机历史仍是外部输入，不在 Session 内采集或同步；
- prompt processor 可先作为外部 preprocessing Region，核心 ABI 只出现
  Tensor/Scalar；
- decode 使用结构化 bounded `for`；
- KV 是 loop-carried SSA/derived cache，不是 authoritative state；
- 同 revision 可 exact reuse，新/缺失 revision 按现有安全语义失效；
- 不引入 action queue；
- 输出组至少包含 `trajectory`、`action_tokens`、`action_token_count`，
  CoT 只能用 bounded token tensor+valid count 表示，不能把 Python string
  放入核心 IR；
- 新增 core op 数目标为 0。

## 7. 完成门槛

### 完整 real L2

- 官方 checkpoint strict load；
- 官方 eager 可重复；
- full-eager capture、Regions、Semantic IR、Plan named outputs 对齐；
- same/new/missing revision；
- failure/abort/retry；
- ResetEpisode；
- core op delta 0。

### 完整 real L3

- 所有 Region 在目标 GPU 重新生成 AOTI package；
- compile report target 与运行 GPU一致；
- 逐 Region和端到端容差在执行前声明；
- 两次 artifact Run 稳定；
- compile/load/first/warm、RSS、CUDA memory、artifact size 完整。

### 完整 real L4

- clean wheel、非 Git cwd；
- verified Compile Bundle；
- generic C ABI 与强类型 wrapper 等价；
- invalid Python 环境仍运行，`ldd` 无 `libpython`；
- schema/hash/shape/dtype/device/layout/target mismatch 均硬失败；
- cache、transaction、reset、failure/retry 和 named outputs 全部通过；
- 至少 1,000 Run soak。

跨 GPU 结果必须单独记录，不能覆盖 RTX 3060 的原论文数据。只完成 A100
不能声明 H20/Orin，反之亦然。

## 8. 结果回传

建议目标机输出：

```text
results/autovla-high-memory/
├── preflight.json
├── run_state.json
├── logs/
├── partition_l2/
├── partition_l3/
├── full_eager/
├── full_l2/
├── full_l3/
└── full_l4/
```

保留 report、manifest、capture、export、artifact、bundle、runner log、`ldd`
和精确命令。回传到 `/home/zhangzimo/Archives/`，不要只留在远端 `/tmp`。

## 9. 可直接用于远端 Codex 的 `/goal`

```text
在 A100 或 H20 CUDA 机器上，基于 codex/vlaforge-paper-artifact 分支继续
完成 AutoVLA 的真实完整 camera-to-trajectory VLAForge 证据。先完整阅读
doc/vlaforge_high_memory_handoff.md，并把其中的 claim boundary 当作硬约束。

第一步运行 vlaforge/tools/run_high_memory_autovla.py --through preflight，
记录 GPU、compute capability、VRAM、Host RAM、磁盘、PyTorch/CUDA/driver，
并验证 AutoVLA source/checkpoint/codebook/Qwen revision 和 SHA256。A100
应原生编译 sm_80，H20 应按实际 capability 原生编译；禁止复用 RTX 3060
sm_86 AOTI package。

随后：
1. 重跑 partition L2→destination-native AOTI→L3 baseline，不修改预声明
   1e-3 Region NRMSE 和 2e-3 trajectory max-abs 门槛。
2. 使用真实、离线、外部已组装的三相机×四帧输入，运行
   probe_real_autovla_full_eager.py，完成官方完整 checkpoint strict load、
   AutoVLA.predict、可捕获路径复跑及真实 prompt/generation/action/trajectory
   capture。
3. 新增 autovla_full Adapter，把处理后视觉 tensor、input_ids/mask、
   ego/route、valid count/mask、InputRevision 声明为外部输入；VLAForge
   不采集/同步传感器。
4. 将完整模型划分为 vision encoder/projector、VLM prefill、bounded
   decode step、action detokenize Regions。KV 使用 loop-carried SSA 或
   derived cache；不得作为 authoritative state；不得引入 action queue。
5. 完成官方 eager、captured Regions、Semantic IR、Plan 的 trajectory、
   action_tokens、token_count 及 bounded CoT token outputs parity，覆盖
   same/new/missing revision、failure/abort/retry、ResetEpisode，core op
   delta 必须为0，达到完整 real L2。
6. 在当前 GPU 逐 Region生成 AOTI package，固定 target/hash/schema/profile，
   预声明数值门槛，完成逐 Region和端到端 parity、重复执行、cold/first/warm、
   Host RSS、CUDA allocated/reserved、compile time和artifact size，达到
   real L3。
7. 生成 verified Compile Bundle、generic C ABI、强类型 C++ wrapper 和
   no-Python Session，完成 schema/hash/shape/dtype/device/layout/target
   负例、typed/generic 等价、cache/transaction/reset/failure/retry、
   1,000 Run soak，达到 real L4。
8. 每个逻辑里程碑独立 commit。大型 checkpoint/capture/artifact 不进 Git，
   统一放持久结果目录，最终回传 /home/zhangzimo/Archives/。fixture、
   部分网络、仅编译未执行的 artifact 不得冒充真实证据。

不得修改冻结 Semantic IR 核心，不得加入 tick/clock/调度、ROS/Cyber、
传感器同步、动作发布、真车或旧 EdgeFM CUDA kernel。最终报告必须区分
当前 A100/H20 correctness/performance 与原 RTX 3060 数据，不外推到未测试
硬件。
```
