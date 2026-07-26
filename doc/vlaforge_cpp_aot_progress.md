# VLAForge v0.2 C++ AOT 进度

> 更新时间：2026-07-25
> 当前语义：独立的 passive `Session::Run()` invocation runtime

## 已完成

### Runtime

- C/C++ Runtime 版本 `0.2.0`；
- `Session::BindTensor/BindScalar/Run/ReadOutput/ResetEpisode`；
- borrowed-until-Run-returns push binding；
- C ABI v2 与 C++ virtual Session API；
- versioned `StateStore`；
- staged transaction commit/abort；
- generic committed named outputs；
- deterministic runtime trace；
- static arena；
- Tensor/Scalar `RegionExecutable` plugin ABI v2。

### Code generation

- 从 Invocation IR v0.2 + physical Plan 生成静态 C++；
- 生成 stable `InputId`/`OutputId`；
- 生成 `ModelInputs`、`ModelOutputs`、`ModelSession`；
- 生成 generic C ABI dispatch；
- embedding `io_schema_digest`；
- required/optional/default input；
- Tensor/Scalar output；
- exact revision cache；
- bounded `if/for`；
- state/output atomic transaction；
- no-Python runner。

### 已覆盖 C++ 场景

1. OpenVLA-like deterministic fixture：
   bounded autoregressive token decode、detokenize、无 queue。
2. SmolVLA-like deterministic fixture：
   prefix condition、flow loop、action chunk queue/cursor 跨 Run。
3. Hybrid driving fixture：
   外部 BEV preprocessing Region、trajectory、agent prediction、
   scalar VQA token、多 named outputs。
4. Driving diffusion fixture：
   two-step denoise、K candidates、scores、selected trajectory 逐元素对齐。
5. C ABI smoke：
   required/optional/default、Tensor+Scalar bind、schema mismatch。
6. Region ABI smoke：
   C customer plugin 使用 Tensor+Scalar value ABI。

### 负契约

- unknown input/output ID；
- required input missing；
- wrong shape/dtype/device/layout/alignment；
- Tensor port 使用 Scalar bind 或反向错误；
- scalar range/valid_count 越界；
- schema digest mismatch；
- binding 未在下一 Run 重新提供；
- validation failure/abort 不推进 state version；
- episode reset。

### 等价性

generated tests 比较：

- Python Semantic Interpreter；
- Python Plan executor；
- generated C++ output；
- normalized input/cache/region/state/transaction/output trace。

runner 在无效 `PYTHONHOME/PYTHONPATH` 下执行，并检查 `ldd` 无 Python。

## Bundle 与 ABI 版本

| Contract | Version |
|---|---|
| Semantic IR | `0.2` |
| I/O schema | `vlaforge.io_schema/2` |
| Region artifact | `vlaforge.region_artifact/3` |
| Region value ABI | `vlaforge.region_executable/2` |
| Session C ABI | `2` |
| Compilation certificate | `vlaforge.compilation_certificate/2` |
| Compile bundle | `vlaforge.compile_bundle/4` |

## JetPack arm64 portability

- `nvcr.io/nvidia/l4t-jetpack:r36.4.0` 在 binfmt 下实际运行
  `aarch64/arm64`；
- 独立 VLAForge Release runtime clean build：18/18；
- arm64 CTest：6/6；
- install/export：runtime library、public headers、CMake targets 完整；
- OpenVLA-like generated Session：arm64 10/10 build + runner passed；
- SmolVLA-like generated Session：arm64 10/10 build + runner passed；
- 两个 runner 均无 Python 动态库依赖。

VLAForge 不依赖任何旧 engine/operator 层。Bundle 只通过
`RegionExecutable` ABI 选择外部 TensorRegion artifact provider；portability
gate 不编译仓库外模型 kernel，也没有旧 CUDA operator fallback。

## 当前证据边界

OpenVLA 与 SmolVLA 已在 2026-07-25 重新通过 v0.2 真实 checkpoint
eager/IR L2；报告位于 `doc/reports/vlaforge_real_v02/`。SmolVLA 又完成真实
artifact L3 与 generated no-Python C++ Session L4。OpenVLA-7B 已通过
36 个 memory-bounded physical Regions 完成真实 `sm_86` artifact L3，
但尚未形成真实 generated Session L4。

因此当前可以声称：

- Invocation IR/Plan/C++ substrate 已贯通；
- 模型范式 fixture 已证明 core expressiveness；
- clean no-Python C/C++ ABI 已验证；
- 固定 `SmolVLA-Base`、DiffusionDrive 与 MindDrive 0.5B checkpoint 已达到
  real Host-CUDA L4；
- OpenVLA-7B 已达到 real Host-CUDA L3；
- robot/flow、robot/autoregressive、driving/diffusion 和
  driving/stateful-multimodal 四种真实模型范式已达到 L3，新增 core op
  均为 0。

当前不能声称：

- OpenVLA 7B 已完成 v0.2 real L4；
- Orin 真机性能或闭环已验证。

## Host CUDA production artifact audit

RTX 3060 12GB、CUDA 12.8、PyTorch 2.10.0+cu128、`sm_86` 已验证：

- `torch.export` → AOTInductor `.pt2`；
- artifact identity、I/O/signature digest、size、SHA-256、ABI、variant、target；
- Compile Bundle v4；
- bundle-relative `create/load/bind/run/synchronize/destroy`；
- generated C++ Session 在无效 `PYTHONHOME/PYTHONPATH` 下执行；
- `ldd` 无 `libpython`；
- eager 与 generated Session 最大绝对误差约 `4.34e-9`；
- missing/corrupt artifact、wrong target、load failure、missing binding、
  output metadata mismatch，以及外部 shape/dtype/device/layout 负例。

该 audit 使用确定性小型 TensorRegion，只证明真实 production backend 通路，
不升级任何 VLA checkpoint 的 L3/L4 等级。

## SmolVLA real Host-CUDA L3/L4

真实 `SmolVLA-Base` 的 prefix、solver-step 与 trim Region 已使用同一
`torch 2.10.0+cu128` 环境编译为 `sm_86` AOTInductor package。10 步
exported pipeline 与 upstream eager 的最终 action bit-exact；artifact
pipeline 在显式 BF16 数值容差内通过，且重复 artifact 执行 bit-exact。详细
hash、size、compile time 和逐步误差见
`doc/reports/vlaforge_real_v03/`。

同一批真实模型 artifacts 已进入八 Region verified Compile Bundle，并由
generated no-Python C++ Session 执行。完整 `[1,50,6]` action chunk 与 direct
AOTI bit-exact；same/new/missing InputRevision、device-resident exact prefix
cache、CUDA authoritative queue/cursor、152 次事务提交、episode reset、
typed/generic C ABI 与 NaN validation abort 均通过。`ldd` 无 `libpython`。
详细证据见 `doc/reports/vlaforge_real_v03/smolvla_artifact_l4.json`。

## OpenVLA-7B real Host-CUDA L3

OpenVLA-7B 的 logical prefill/decode/detokenize 被细化为 36 个 backend-owned
physical Regions，以 two-layer chunk 在 RTX 3060 12GB 上完成 capture、
active-version normalization、`sm_86` AOTI compile 和 artifact-only
autoregressive pipeline。KV 使用固定 `[1,32,281,128]` buffer、显式
`cache_position` 和 loop-carried SSA，是 140.5 MiB derived cache，不是
persistent state。

26.316 GiB artifacts 的逐 Region 最大 NRMSE 为 `0.02688469`，integer/token
输出 exact；两次完整 pipeline 的 7 个 action token 与最终 action
bit-exact，且与真实 L2 action 的最大绝对误差仅 `1.13e-17`。报告来自 clean
revision `7ea773e`，core op delta 为 0。详细证据见
`doc/reports/vlaforge_real_v03/openvla_artifact_l3.json`。

这是 real L3，不是 L4。当前 eager-load generated Session 会同时常驻所有
Region weights，不适合 12GB GPU；下一步只通过 generic artifact residency
policy 尝试 weight paging，不把模型专属路由写入 core。

## 当前完成边界与可选增强

本机必选工作已经完成：SmolVLA/DiffusionDrive real L4 的 Host-CUDA
latency、memory、profile、四类正式消融和 10k Run 长稳均有正式报告；
MindDrive 0.5B 的 8 logical/66 physical artifact、16-state、10-output
generated no-Python C++ real L4 correctness 已闭合；冻结 core 后的 Octo、
GR00T N1.7、AutoVLA held-out 审计均为 core-op delta 0；AutoVLA 又完成
发布 checkpoint 的真实 L2 decoder partition。

以下只属于可选增强，不保持当前论文 Goal 未完成：

1. 以稳定 extracted-library/cubin provider 替代当前 PyTorch package-loader，
   再尝试 OpenVLA generated no-Python C++ L4；现有资源 blocker 已正式记录；
2. 扩展 AutoVLA 到 camera/prompt/VLM prefill 和完整 autoregressive decode，
   并在不放宽预声明数值门槛的前提下争取 real L3/L4；
3. 增加第二机或其他 GPU 的独立复现；
4. JetPack r36.4 ARM64 Docker 已完成 TensorRT Region backend、installed
   SDK consumer、generated TensorRT Session 和 identity-engine 上板 smoke
   的编译；Orin 台架就绪后执行 smoke，并可选补充真实模型 SM87
   parity/latency/power/thermal。

## 2026-07-27 Host-CUDA release audit

- offline Python suite：263 passed，11 个 opt-in gate skipped；
- real SmolVLA checkpoint gate：1 passed；
- real SmolVLA L4 opt-in gate：1 passed；
- real OpenVLA-7B 4-bit gate：1 passed；
- real OpenVLA-7B partitioned artifact L3：36/36 Regions 与两次完整
  pipeline passed；
- real MindDrive 0.5B generated L4：66 physical artifacts、10 outputs、
  typed/generic/compiled-reference exact、trace/failure/reset passed；
- clean C++ Release build：passed；
- CPU CTest：7/7 passed；CUDA/AOTI CTest：8/8 passed；
- CPU 与 AOTI CMake install/export consumer：passed；
- wheel：`vlaforge-0.2.0.dev0-py3-none-any.whl` built；
- arm64 JetPack image probe：`aarch64`；
- arm64 standalone runtime：18/18 build、6/6 CTest、install/export passed；
- arm64 OpenVLA-like/SmolVLA-like generated Session：build/run passed；
- `git diff --check`：passed。

机械完成审计见
`doc/reports/vlaforge_paper_completion_v01/paper_completion.json`，当前为
`submission_ready=true`。该状态只覆盖 RTX 3060 `sm_86` / CUDA 12.8，
不外推为跨 GPU 或 Orin 性能结论。
