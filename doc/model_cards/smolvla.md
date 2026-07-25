# SmolVLA Model Adaptation Card

| 项 | 当前记录 |
|---|---|
| Upstream | `huggingface/lerobot@8fff0fde7c79f23a93d845d1a50e985de01f8b8a` |
| License / checkpoint | source/VLM 为 Apache-2.0；policy 本地目录无独立 model card；checkpoint SHA256 `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb` |
| Source entry | `policies/smolvla/modeling_smolvla.py`、`configuration_smolvla.py` |
| 当前证据 | L0 + L1 + real L2 + real Host-CUDA L3 + real Host-CUDA L4 |
| Adapter | fixture 与 real capture/audit 独立于 core；最终 LOC 在论文冻结时重算 |
| Core op 增量 | 0 |

authoritative state 为 Adapter 侧 `action_queue`、`queue_cursor`、RNG；prefix 是
derived exact cache；四步 flow loop 使用 loop-carried SSA。C++ fixture 连续
Run 验证 queue refill/consume、成功 commit 版本递增、episode reset 和 typed/C
ABI 输出一致。

2026-07-25 已在 RTX 3060 上运行本地 SmolVLA checkpoint：10 个 solver step、
最终 action 和 3 次 queue output 均 `max_abs_error=0`，证据见
`doc/reports/vlaforge_real_v02/smolvla_eager_ir.json`。这是真实 eager ↔
Invocation IR L2。单次审计 peak 约 920.47 MiB，不是稳定 benchmark。

2026-07-25 进一步在 `torch 2.10.0+cu128` 上将真实 prefix、solver-step 和
trim Region 编译为 `sm_86` AOTInductor package。exported 10 步 pipeline 与
upstream eager 的最终 action bit-exact；artifact 与 eager 的最终 action
`max_abs=0.02784944`、`mean_abs=0.00802071`、`NRMSE=0.01303127`，重复
artifact 执行 bit-exact。该结果按显式 BF16 数值容差判为 real L3，不声称
exact parity。证据见
`doc/reports/vlaforge_real_v03/smolvla_artifact_l3.json`。

同日完成真实 L4：上述三个真实模型 artifact 与五个 Adapter-owned support
Region 进入同一 verified Compile Bundle，由 generated no-Python C++ Session
在 RTX 3060 上执行。direct AOTI 与 C++ 的完整 `[1,50,6]` action chunk
bit-exact；连续 152 次 Run 验证 queue consume/refill、same-revision cache
hit、new/missing-revision miss、CUDA authoritative state version、episode
reset、typed/generic C ABI，以及 NaN validation failure 的事务 abort。runner
在无效 `PYTHONHOME/PYTHONPATH` 下运行，`ldd` 无 `libpython`。证据见
`doc/reports/vlaforge_real_v03/smolvla_artifact_l4.json`。
