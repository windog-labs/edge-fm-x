# EdgeFM 文档索引

`doc/` 只保留面向用户和评审者的设计文档、平台说明、持续维护的调优结论，以及需要长期保留的评审记录。一次性 debug 记录、临时调优草稿和已经吸收到正式文档中的 scratch notes 不在这里长期保留。

## 核心文档

- `doc/design.md` - 代码结构、运行时分层、配置与调度
- `doc/edge_fm_benchmark_tables.md` - 当前维护中的 benchmark 数据表
- `doc/orin_r36.4.3_qwen_benchmark_guide.md` - Jetson Orin benchmark 指南
- `doc/smolvla_phase1_horizon_usage.md` - SmolVLA phase-1 Horizon 双 stage 导出与 `prefill`/`decode` 调用示例
- `README.md` - 仓库总览、支持模型、平台状态与使用入口

## RTX 3060 LLM 调优

- `doc/3060_tuning_rules.md` - 3060 调优的固定规则与边界
- `doc/3060_tuning_plan.md` - 当前有效目标、活跃实验队列与验收标准
- `doc/3060_tuning_log.md` - 最新 baseline、接受/拒绝结论、raw artifact 与 profiling 结论

3060 当前的结论已经更新：Stage 1 已关闭，Qwen2.5 内部 TensorRT engine prefill bridge 不再作为 EdgeFM 代码路径维护；主线是 `EdgeFM(cuda graph)` + source-op CUTLASS/CUDA operator。当前 source-op 通过通用 `linear` / `mlp` layer-operator 边界和 3060 operator table 选择，不再由 `qwen2_5.cpp` 直接调用模型私有 bridge。外部 `TRT-Edge-LLM` 仍作为 Stage 2 benchmark reference；source-visible/plugin-op 资产可以继续评估，但不能依赖 serialized TensorRT engine bridge。

最新 3060 LLM 全矩阵为 `Qwen2.5-{0.5B,1.5B,3B}` ×
`prefill={512,1024,2048}` × `decode={32,64}`：`16/18` 个 shape 快于
`TRT-Edge-LLM`，0.5B 和 3B 全 shape 快于 TRT reference，唯一稳定正差距是
`1.5B 512x64` 约 `+0.9 ms`。`1.5B 512x32` 在高 runs 复核中已接近测量噪声
（avg `+0.197 ms`，median `+0.135 ms`）。

## 当前代码结构速览

- `src/engine/`：EdgeFM facade、`EngineConfig`、`EngineFactory`，以及按 task
  分组的 engine。
- `src/engine/tasks/token_generation/`：LLM/VLM token 生成路径，包含
  `KVManager`、scheduler、compact vocab、CUDA 标准引擎相关状态。
- `src/engine/tasks/trajectory_planning/`：trajectory planner policy 路径，
  包含 `TrajectoryPlannerEngine`、`PlannerStateManager` 和 tensor 工具。
- `src/engine/tasks/stage_execution/`：命名 stage 入口，当前保留 mock runner
  和 stage_execution engine。
- `src/backends/`：backend artifact/cache/runtime 边界，当前主要承载 Horizon
  artifact 与 runtime metadata。
- `src/layers/`：模型层语义与权重组织，例如 linear、attention、gated MLP。
- `src/operators/`：operator registry、operator impl table、CUDA/CUTLASS/
  FlashInfer/source-op 具体实现。

Qwen2.5 的 3060 source-op 路径已经从模型私有 bridge 抽到
`src/operators/prefill_linear_source_op.*` 和
`src/operators/prefill_mlp_source_op.*`，由 3060 operator table 选择。保留在
`src/python/pybind_trt_runtime.cpp` 里的 `edge_fm_trt` 是 benchmark/reference
入口，不是默认 EdgeFM generate 路径。

## 3060 评审记录

- `doc/3060_fused_mlp_review.md` - 3060 prefill MLP / bridge 相关评审与拒绝结论
- `doc/3060_qkv_oproj_bridge_review.md` - 3060 QKV / OProj TensorRT bridge 评审记录
