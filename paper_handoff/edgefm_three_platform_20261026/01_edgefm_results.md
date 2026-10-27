# EdgeFM 框架指标与baseline对应关系

**框架指标已经实测，不是只跑了原始模型。模型名称是工作负载，方法列才区分baseline和EdgeFM。**

**但这不是五模型统一启用全部Agent算子优化+runtime优化的“Full EdgeFM”结果。** 当前完整模型主表验证框架部署与所列runtime模式；算子微基准/回集成单列。Norm回集成负收益，GEMM/Attention未过完整动作门槛，不能视为已组合启用的有效收益。

| 平台 | 原表标签 | 实际测量对象 |
|---|---|---|
| H20 | official-pytorch | 官方PyTorch原始实现baseline |
| H20 | session-required | EdgeFM生成的native C++ Session，完整solver-loop replay；外围原始输入工作流仍有Python |
| Orin | Conventional TRT / Ordinary repeated-call TRT | 同一EdgeFM生成的native Session关闭相关优化，普通TRT执行；不是独立裸trtexec |
| Orin | Shared stream / Full-loop replay | EdgeFM共享流或完整solver-loop CUDA Graph replay |
| Orin | Verified arena; no replay | EdgeFM静态Arena布局优化，未实现/未宣称replay |

## 已实测的完整VLA对照

| 平台 | 工作负载 | baseline mean ms | EdgeFM mean ms | 均值比 | EdgeFM方法 |
|---|---|---:|---:|---:|---|
| H20 | SmolVLA | 206.161639 | 50.726747 | 4.064160x | EdgeFM native Session + full-loop replay |
| H20 | RDT-1B | 239.685959 | 221.804152 | 1.080620x | EdgeFM native Session + full-loop replay |
| H20 | pi0 | 221.305113 | 108.152217 | 2.046237x | EdgeFM native Session + full-loop replay |
| H20 | pi0.5 | 256.840307 | 116.853524 | 2.197968x | EdgeFM native Session + full-loop replay |
| H20 | CogACT (public dependencies) | 120.286270 | 79.971651 | 1.504111x | EdgeFM native Session + full-loop replay |
| Orin | SmolVLA | 134.409630 | 113.139929 | 1.187995x | EdgeFM + TRT full-loop replay |
| Orin | RDT-1B | 3065.625218 | 2975.587343 | 1.030259x | EdgeFM + TRT full-loop replay |
| Orin | pi0 | 1129.869939 | 1125.379221 | 1.003990x | EdgeFM + TRT verified Arena (no replay) |
| Orin | pi0.5 | 1258.792712 | 1256.955605 | 1.001462x | EdgeFM + TRT verified Arena (no replay) |
| Orin | CogACT (actual ~7.63B public-dependency configuration) | 694.393388 | 693.790506 | 1.000869x | EdgeFM + TRT verified Arena (no replay) |

**边界：**Orin表衡量同一Session中关闭/启用优化的增益，不是EdgeFM对TensorRT-Edge-LLM或独立裸trtexec的外部baseline优越性。pi0/pi0.5/CogACT仅Arena，无replay，差异很小，不宣称显著增益。
H20与Orin baseline/精度/边界不同，不跨平台相除。Qwen native部署也是框架适配指标，但H20的first/full固定profile不是同请求逐token计时。

[完整三平台主表、CDF、质量和消融](02_all_platform_results.md)；[框架对照CSV](tables/edgefm_framework_comparison.csv)。
保留旧configuration/mode列作溯源；新增paper_method列只澄清命名，不产生新实验数据。
