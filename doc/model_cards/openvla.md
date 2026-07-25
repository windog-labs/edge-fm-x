# OpenVLA Model Adaptation Card

| 项 | 当前记录 |
|---|---|
| Upstream | `openvla/openvla@c8f03f48af692657d3060c19588038c7220e9af9` |
| License / checkpoint | MIT；`openvla/openvla-7b` 由环境指定，未在 bundle 固定 hash |
| Source entry | `modeling_prismatic.py`、`experiments/robot/openvla_utils.py` |
| 当前证据 | L0 + L1 + real L2 + deterministic fixture-L4 |
| Adapter | fixture 196 LOC；real adapter 598 LOC（含 capture/audit） |
| Core op 增量 | 0 |

静态 image + instruction 输入；prefill/context Region 使用 exact cache，action
token decode 是 bounded loop-carried SSA，随后 detokenize。没有 action queue
或 control epoch。生成 C++ fixture 同时验证 typed wrapper、generic C ABI、
schema digest 和无 Python 链接。

2026-07-25 已在 RTX 3060 上运行本地 OpenVLA-7B 4-bit checkpoint：
7 个 action token 与 action 全量精确相等，`action_max_abs_error=0`，
证据见 `doc/reports/vlaforge_real_v02/openvla_eager_ir.json`。这是 eager ↔
Invocation IR 的真实 L2，不是 compiled artifact。

`fixture-L4` 不是 7B checkpoint 的 L4。本轮仍未形成 v0.2 real
artifact/no-Python Session 证据。真实单次审计记录 peak 约 4756.77 MiB；
该数值不是稳定 latency benchmark。
