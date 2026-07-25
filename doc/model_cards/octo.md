# Octo Model Adaptation Card

| 项 | 当前记录 |
|---|---|
| Upstream | `octo-models/octo@241fb3514b7c40957a86d869fecb7c7fc353f540` |
| License / checkpoint | MIT；`rail-berkeley/octo-base-1.5` 未下载、hash 未固定 |
| Source entry | `OctoModel.sample_actions`、`DiffusionActionHead` |
| 当前证据 | L0 + L1 deterministic fixture |
| Adapter | `build_octo_like_fixture`，180 LOC |
| Core op 增量 | 0 |

输入包含 observation history，以及有 default 的 optional language/goal-image
端口；输出整段 action chunk。`octo_condition` 是 exact derived cache，三步
denoise 使用 bounded for 与 loop-carried SSA。真实 Octo 默认 diffusion head
可配置约 20 步，当前三步仅是结构验证，不是数值复现。

不支持项：JAX frontend capture、checkpoint parity、artifact/C++ parity。
Memory/performance：pending。
