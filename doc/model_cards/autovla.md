# AutoVLA Model Adaptation Card

| 项 | 当前记录 |
|---|---|
| Upstream | `ucla-mobility/AutoVLA@ba34eed74ce6729e7986592d0e66cbaca397b4fa` |
| License | UCLA Academic Software License；仅按研究用途使用 |
| Checkpoint | `Zewei-Zhou/AutoVLA@a7d7ba3ed7529b248d2694c2defa31b35208340f`，`AutoVLA_PDMS_89.ckpt`，16,292,664,780 bytes，SHA256 `58246773393da45678a3f35d354fd969eed6833ecc8ee596edc5e283d1a87473` |
| Qwen base config | `Qwen/Qwen2.5-VL-3B-Instruct@66285546d2b821cf421d4f5eb2576359d3770cd3` |
| Source entry | `models/autovla.py:AutoVLA.predict`、`models/action_tokenizer.py` |
| 当前证据 | pinned L0 + executable L1 + `L2-partitioned-real-checkpoint-frontend` |
| Adapter | fixture `build_driving_ar_fixture`；real `autovla_real.py` |
| Core op 增量 | 0 |

## 真实 L2 分区

真实 checkpoint 在 RTX 3060/CUDA 12.8 上以 mmap 加载，只选择 Qwen 第 35
层 post-attention RMSNorm+MLP、final RMSNorm、真实 action-vocabulary
projection rows，以及发布的 2,048-entry vehicle codebook。对外输入是已经由
上游有界 decode 准备好的 `[1,10,2048]` BF16 post-attention hidden tensor；
输出组原子提交：

- `trajectory [10,3]`；
- `action_tokens [10]`。

三个 TensorRegion 分别是 `autovla_decoder_mlp`、
`autovla_action_projection` 和 `autovla_trajectory_decode`。真实 eager、
strict export、Semantic IR 和 Plan 的 trajectory/token 均 exact，
Semantic/Plan trace exact。revision 序列 `[100,100,101]` 产生 1 hit/2 miss，
证明 held-out 真实模型复用现有 InputRevision cache 语义。verified static
arena 为 123,200 bytes；单次审计 peak CUDA allocated 533,944,320 bytes，
peak Host RSS 1,473,228,800 bytes。这些是 correctness-audit envelope，不是
性能 benchmark。

正式报告：
`../reports/vlaforge_autovla_v01/autovla_frontend_l2.json`。

## L3 有界尝试

默认与预定义 conservative AOTI profile 都成功编译三个 `sm_86` package。
conservative 结果的 10 个 action token exact，trajectory 最大绝对误差
`1.91e-6`，重复执行所有输出 bit-exact；但 decoder hidden 和 action logits
NRMSE 分别为 `6.65e-3` 与 `4.54e-3`，超过预声明 `1e-3` Region 门槛。因此
该结果严格保留为 `L3-candidate`，不升级为 real L3，也没有事后放宽阈值。
报告见
`../reports/vlaforge_autovla_v01/autovla_artifact_l3_candidate.json`。

## 边界与不支持项

该 L2 是真实权重 decoder 分区，不是完整端到端 AutoVLA：

- 不包含 camera decode、传感器同步或输入采集；
- 不包含 prompt/processor、vision encoder、prefill/attention；
- 不包含完整 autoregressive token generation 和 fast/slow CoT；
- 不包含 generated no-Python C++ Session；
- 不声明真实规划闭环、真车或 Orin 性能。

fixture 仍用于 fast/slow branch 和 bounded token-decode 的结构覆盖，不能替代
上述真实 checkpoint 证据。冻结核心审计见
`../reports/vlaforge_heldout_v01/heldout_audit.md`。
