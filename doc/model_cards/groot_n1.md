# GR00T N1.7 Model Adaptation Card

| 项 | 当前记录 |
|---|---|
| Upstream | `NVIDIA/Isaac-GR00T@9c7e746b2cd37a810070a98ef41d290a07e806c2` |
| License / checkpoint | Apache-2.0 code；weights 为 NVIDIA Open Model License；未下载 |
| Source entry | `gr00t_n1d7.py`、`modules/dit.py`、`gr00t_policy.py` |
| 当前证据 | L0 + L1 deterministic fixture |
| Adapter | `build_groot_n1_like_fixture`，185 LOC |
| Core op 增量 | 0 |

多相机、robot state、optional language 与 bounded `embodiment_id` 都是静态
InputPort。VLM prefix 与 embodiment logits 是一组 artifact/derived cache，
action expert 是四步 bounded DiT loop；输出 action chunk 与 aux logits 两个
named outputs。TensorRT/custom backend 通过 artifact/Region extension 表达，
没有模型专属 core op。

不支持项：真实 N1.7 capture、TensorRT artifact、C++ parity。Memory/performance：
pending。
