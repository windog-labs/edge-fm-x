# OpenVLA Model Adaptation Card

| 项 | 当前记录 |
|---|---|
| Upstream | `openvla/openvla@c8f03f48af692657d3060c19588038c7220e9af9` |
| License / checkpoint | MIT；`openvla/openvla-7b@47a0ec7fc4ec123775a391911046cf33cf9ed83f`；三个 shard SHA256 已固定在 L3 报告 |
| Source entry | `modeling_prismatic.py`、`experiments/robot/openvla_utils.py` |
| 当前证据 | L0 + L1 + real L2 + real L3 + deterministic fixture-L4 |
| Adapter | fixture 222 LOC；real semantic adapter 599 LOC；frontend 607 LOC；L3 physical partition adapter 980 LOC |
| Core op 增量 | 0 |

静态 image + instruction 输入；prefill/context Region 使用 exact cache，action
token decode 是 bounded loop-carried SSA，随后 detokenize。没有 action queue
或 control epoch。生成 C++ fixture 同时验证 typed wrapper、generic C ABI、
schema digest 和无 Python 链接。

2026-07-25 已在 RTX 3060 上运行本地 OpenVLA-7B 4-bit checkpoint：
7 个 action token 与 action 全量精确相等，`action_max_abs_error=0`，
证据见 `doc/reports/vlaforge_real_v02/openvla_eager_ir.json`。这是 eager ↔
Invocation IR 的真实 L2，不是 compiled artifact。

真实 L3 将相同逻辑细化成 36 个 backend-owned physical Regions：multimodal
prepare、16 个 two-layer prefill chunks、token embedding、16 个 fixed-KV
decode chunks、logits head 和 detokenize。固定 KV profile 为 prefix 275、
最大长度 281、6 次 bounded decode；64 个 KV tensor 共 140.5 MiB，属于
loop-carried derived cache，不是 Session persistent state。

36 个 export 在当前 PyTorch 版本重导出后逐输出保持 exact，随后编译成
26.316 GiB `sm_86` AOTInductor packages。逐 Region artifact 最大 NRMSE
为 `0.02688469`，低于预先固定的 `0.05` BF16 contract；integer/token
输出 exact。两次完整 artifact-only pipeline 都生成 token
`[31857, 31864, 31900, 31840, 31860, 31868, 31872]`，最终 action 相对
L2 reference 最大绝对误差 `1.13e-17`，两次执行 bit-exact。

证据见 `doc/reports/vlaforge_real_v03/openvla_artifact_l3.json`。该报告来自
clean revision `7ea773e`，新增 core op 为 0。capture/audit 峰值 CUDA
allocated 分别为 2.686/1.778 GiB，单 Region 编译峰值 host RSS 为
6.246 GiB；这些是正确性审计元数据，不是 paper-grade latency benchmark。

`fixture-L4` 仍不是 7B checkpoint 的 L4。真实 generated no-Python C++
Session 需要支持受验证的 weight-paged Region residency，当前保持
`unsupported`，不得把 real L3 写成 real L4。
