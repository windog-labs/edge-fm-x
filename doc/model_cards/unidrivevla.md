# UniDriveVLA Model Adaptation Card

| 项 | 当前记录 |
|---|---|
| Upstream | `xiaomi-research/unidrivevla@a93c175af893b35dc16618e659eca4d18bb1ec86` |
| License / checkpoint | 根目录 license 需进一步法务确认；checkpoint 未下载 |
| Source entry | detector、Qwen3-VL planning head、unified perception decoder |
| 当前证据 | source-tree audit L0 + structural L1 |
| Adapter | 复用 HybridExternalFeature 180 LOC；专属 adapter pending |
| Core op 增量 | 0 |

2D/3D perception tokens、map/agent features 与多专家路由优先封装在
TensorRegion；跨 artifact 选择才使用结构化 branch/variant。检测、地图、
轨迹、aux task 都映射为 generic named output group，不引入 action queue。
bounded agent/map 数量通过 max shape + valid count/mask。

不支持项：license 确认、真实 capture、专家 artifact、C++ parity。
Memory/performance：pending。
