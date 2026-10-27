# π0 Model Adaptation Card

> Historical L0/L1 fixture card, not the real checkpoint/deployment profile.
> Its four-step fixture below must not replace the ten-step OpenPI workload.
> Current source/data identities are in the [2026-10-22 asset assessment](../reports/orin_source_asset_assessment_20261022.md)
> and [real H20 variants](../reports/pi0_execution_variants_20260908.md).
> Orin deployment/compile/output acceptance is separate and still pending.

| 项 | 当前记录 |
|---|---|
| Upstream mapping | `huggingface/lerobot@0d383d09f2051444de211739196a28cc94736861` |
| License / checkpoint | Apache-2.0；未固定 π0 checkpoint |
| Source entry | `policies/pi0/modeling_pi0.py`、`configuration_pi0.py` |
| 当前证据 | L0 + L1 deterministic fixture |
| Adapter | `build_pi0_fixture`，203 LOC |
| Core op 增量 | 0 |

VLM prefix 是 exact derived cache；proprio 和 initial noise 进入四步 bounded
flow loop；模型输出完整 continuous action chunk。fixture 不在 core 中保留
queue，底软可直接消费整段输出；若需要逐步消费，可复用 ChunkedAction Adapter。

不支持项：真实 checkpoint/capture/artifact/C++。Memory/performance：pending。
