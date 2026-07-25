# ACT Model Adaptation Card

| 项 | 当前记录 |
|---|---|
| Upstream | `tonyzhaozh/act@742c753c0d4a5d87076c8f69e5628c79a8cc5488` |
| License / checkpoint | MIT；未下载真实 checkpoint |
| Source entry | `policy.py`、`detr/models/detr_vae.py` |
| 当前证据 | L0 + L1 deterministic fixture |
| Adapter | `build_act_like_fixture`，167 LOC |
| Core op 增量 | 0 |

`act_predict_chunk` 生成整段 action chunk。`action_queue` 与 `queue_cursor`
是 `ChunkedAction` Adapter 的 authoritative state，通过
read-latest → stage-write → commit 跨 Run 消费；它们不是 core IR 假设。
分支只决定 refill/reuse，输出仍是通用 `robot_action` output group。

不支持项：真实 ACT checkpoint/capture/artifact/C++。Memory/performance：
pending。
