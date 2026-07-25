# AutoVLA Model Adaptation Card

| 项 | 当前记录 |
|---|---|
| Upstream | `ucla-mobility/AutoVLA@ba34eed74ce6729e7986592d0e66cbaca397b4fa` |
| License / checkpoint | UCLA Academic Software License；checkpoint 未下载 |
| Source entry | `models/autovla.py:AutoVLA.predict`、`models/action_tokenizer.py` |
| 当前证据 | pinned upstream Git-object L0 + verified executable L1 |
| Adapter | `build_driving_ar_fixture`，240 LOC |
| Core op 增量 | 0 |

源码路径以 VLM `generate` 输出 action tokens，按固定 trajectory pose 数截断/
补齐后 detokenize；文本可携带 fast/slow CoT。fixture 用结构化 `if` 表达
fast/slow artifact routing，用 bounded for 表达 trajectory-token decode，
最后输出轨迹。CoT 文本本身仍是 TensorRegion 内部 token graph，不加入新 op。

冻结 core 后，fast 与 slow 路径共 3 次 fixture Run 的 Semantic/Plan output、
state 和完整 trace exact；prefill cache 为 1 hit/2 miss。verified static
arena 从 640 bytes 降到 256 bytes。上游 `predict`、VLM `generate`、bounded
token 截断/补齐和 trajectory detokenize 均由 pinned Git objects 验证。

不支持项：真实 checkpoint、真实 fast/slow parity、artifact/C++ parity。
商业使用前需单独审查 upstream academic-only license。
冻结审计见 `../reports/vlaforge_heldout_v01/heldout_audit.md`。
