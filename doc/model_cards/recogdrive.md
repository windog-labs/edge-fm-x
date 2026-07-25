# ReCogDrive Model Adaptation Card

| 项 | 当前记录 |
|---|---|
| Upstream | `xiaomi-research/recogdrive@d54404796de7a44ca418b96057e3f8c3de3e8c0d` |
| License / checkpoint | Apache-2.0；checkpoint 未下载 |
| Source entry | `recogdrive_agent.py`、`recogdrive_diffusion_planner.py`、`recogdrive_dit.py` |
| 当前证据 | 实际 source audit L0 + structural L1 |
| Adapter | 复用 HybridExternalFeature 180 LOC；专属 real adapter 尚未实现 |
| Core op 增量 | 0 |

源码审计确认 VLM hidden state 与 history/status 进入 diffusion planner，默认
action horizon 为 8、inference steps 为 5；这应划分为 VLM artifact、condition
tensor、bounded DiT/flow Region。跨 artifact 的数据仍是静态 Tensor ABI。

当前 Hybrid fixture 只验证这种跨 artifact、多 named output 结构，不是
ReCogDrive 数值 fixture。未完成 real capture/artifact/C++ parity。
Memory/performance：pending。
