# SmolVLA Model Adaptation Card

| 项 | 当前记录 |
|---|---|
| Upstream | `huggingface/lerobot@0d383d09f2051444de211739196a28cc94736861` |
| License / checkpoint | Apache-2.0；`lerobot/smolvla_base` 未在 bundle 固定 hash |
| Source entry | `policies/smolvla/modeling_smolvla.py`、`configuration_smolvla.py` |
| 当前证据 | L0 + L1 + real L2 + deterministic fixture-L4 |
| Adapter | fixture 311 LOC；real adapter 743 LOC（含 capture/audit） |
| Core op 增量 | 0 |

authoritative state 为 Adapter 侧 `action_queue`、`queue_cursor`、RNG；prefix 是
derived exact cache；四步 flow loop 使用 loop-carried SSA。C++ fixture 连续
Run 验证 queue refill/consume、成功 commit 版本递增、episode reset 和 typed/C
ABI 输出一致。

2026-07-25 已在 RTX 3060 上运行本地 SmolVLA checkpoint：10 个 solver step、
最终 action 和 3 次 queue output 均 `max_abs_error=0`，证据见
`doc/reports/vlaforge_real_v02/smolvla_eager_ir.json`。这是真实 eager ↔
Invocation IR L2。单次审计 peak 约 920.47 MiB，不是稳定 benchmark。

`fixture-L4` 不代表真实 SmolVLA checkpoint 已经 L4。真实 v0.2 artifact 与
no-Python Session parity 仍 pending。
