# VLAForge v0.2 C++ AOT 进度

> 更新时间：2026-07-25
> 当前语义：passive `Session::Run()`；本文不再记录 v0.1 tick/epoch runner

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
- Python Scheduled Plan executor；
- generated C++ output；
- normalized input/cache/region/state/transaction/output trace。

runner 在无效 `PYTHONHOME/PYTHONPATH` 下执行，并检查 `ldd` 无 Python。

## Bundle 与 ABI 版本

| Contract | Version |
|---|---|
| Semantic IR | `0.2` |
| I/O schema | `vlaforge.io_schema/2` |
| Region artifact | `vlaforge.region_artifact/2` |
| Region value ABI | `vlaforge.region_executable/2` |
| Session C ABI | `2` |
| Compilation certificate | `vlaforge.compilation_certificate/2` |
| Compile bundle | `vlaforge.compile_bundle/3` |

## 已清理的 v0.1 路径

以下内容不再属于 production：

- `Epoch`/`EpochCache`；
- `ActionQueue::Publish`；
- `RunTick`；
- epoch-keyed memoization；
- tick-based real AOTI/TorchScript runner；
- v0.1 whole-program benchmark。

Git 历史和 `doc/reports/` 中的旧报告只作为历史证据，不可作为 v0.2 release
结果。

## 当前证据边界

OpenVLA 与 SmolVLA 已在 2026-07-25 重新通过 v0.2 真实 checkpoint
eager/IR L2；报告位于 `doc/reports/vlaforge_real_v02/`。本轮 v0.2
generated C++ 证据仍是 deterministic `fixture-L4`，真实 checkpoint 的
artifact + generated Session L3/L4 需要按新 ABI 重建。

因此当前可以声称：

- Invocation IR/Plan/C++ substrate 已贯通；
- 模型范式 fixture 已证明 core expressiveness；
- clean no-Python C/C++ ABI 已验证。

当前不能声称：

- OpenVLA 7B 或 SmolVLA checkpoint 已完成 v0.2 real L4；
- DiffusionDrive checkpoint 已运行 generated C++；
- Orin 真机性能或闭环已验证。

## 下一步

1. 以新 Session generator 接回 real AOTI/TensorRT Region artifact；
2. 先完成 DiffusionDrive、SmolVLA、OpenVLA 的真实 L3/L4；
3. Host CUDA latency/memory/profile 与长稳；
4. Orin 环境就绪后 arm64 clean build 与真机测量。

## 2026-07-25 Release audit

- offline Python suite：169 passed，3 个 opt-in gate skipped；
- real SmolVLA checkpoint gate：1 passed；
- real OpenVLA-7B 4-bit gate：1 passed；
- clean C++ Release build：passed；
- CTest：6/6 passed；
- CMake install/export：passed；
- wheel：`vlaforge-0.2.0.dev0-py3-none-any.whl` built；
- arm64 JetPack image probe：`aarch64`；
- `git diff --check`：passed。
