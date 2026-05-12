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

3060 当前的结论已经收敛：主线仍是 `EdgeFM(cuda graph)`，但针对 prefill 的缩小差距路线是默认关闭、可回退的 TensorRT `subgraph / subengine bridge`。原因不是 EdgeFM 整体设计失效，而是 `nsys` 已确认 TRT-Edge-LLM 在该平台上大量使用 TensorRT 编译器生成的闭源 `Myelin / XMMA / FcCast` 类内核，这些能力无法直接作为普通 EdgeFM operator 复用。

## 3060 评审记录

- `doc/3060_fused_mlp_review.md` - 3060 prefill MLP / bridge 相关评审与拒绝结论
- `doc/3060_qkv_oproj_bridge_review.md` - 3060 QKV / OProj TensorRT bridge 评审记录
