# GR00T N1.7 Model Adaptation Card

| 项 | 当前记录 |
|---|---|
| Upstream | `NVIDIA/Isaac-GR00T@9c7e746b2cd37a810070a98ef41d290a07e806c2` |
| License / checkpoint | Apache-2.0 code；weights 为 NVIDIA Open Model License；未下载 |
| Source entry | `gr00t_n1d7.py`、`modules/dit.py`、`gr00t_policy.py` |
| 当前证据 | pinned upstream Git-object L0 + verified executable L1 |
| Adapter | `build_groot_n1_like_fixture`，186 LOC |
| Core op 增量 | 0 |

多相机、robot state、optional language 与 bounded `embodiment_id` 都是静态
InputPort。VLM prefix 与 embodiment logits 是一组 artifact/derived cache，
action expert 是四步 bounded DiT loop；输出 action chunk 与 aux logits 两个
named outputs。TensorRT/custom backend 通过 artifact/Region extension 表达，
没有模型专属 core op。

冻结 core 后，2 次 fixture Run 的 Semantic/Plan output、state 和完整 trace
exact；prefix cache 为 1 hit/1 miss。verified static arena 从 640 bytes
降到 320 bytes。上游 action head、embodiment-conditioned encoder、
bounded inference loop 和 DiT timestep encoder 均由 pinned Git objects
验证。

不支持项：真实 N1.7 capture、TensorRT artifact、C++ parity。Memory/performance：
pending。冻结审计见
`../reports/vlaforge_heldout_v01/heldout_audit.md`。
