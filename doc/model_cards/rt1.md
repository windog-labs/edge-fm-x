# RT-1 Model Adaptation Card

| 项 | 当前记录 |
|---|---|
| Upstream | `google-research/robotics_transformer@4569641b8111f3f402c32d8e24becd2a6e952ecc` |
| License / checkpoint | Apache-2.0；未下载真实 checkpoint |
| Source entry | `robotics_transformer/model.py` |
| 当前证据 | L0 source contract + L1 deterministic fixture |
| Adapter | `build_rt1_like_fixture`，108 LOC |
| Core op 增量 | 0 |

输入是固定上界的短图像/观测 history、`history_valid_mask` 和 language tokens；
输出组同时包含离散 `action_token` 与 detokenized continuous `action`。模型无跨
Run authoritative state；history 由底软组装。Region 划分为
`rt1_action_token` 与 `rt1_detokenize`，无循环、无内部调度、无跨 Run cache。

不支持项：真实权重 capture、真实 artifact、无 Python C++ parity。当前 fixture
只证明 history+mask、离散 token 与 detokenize 能由现有 IR 表达。
Memory/performance：pending。
