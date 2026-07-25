# OpenDriveVLA Model Adaptation Card

| 项 | 当前记录 |
|---|---|
| Upstream | `DriveVLA/OpenDriveVLA@10e8095bc618d508cb70cca37b6956ac4db6e9f3` |
| License / checkpoint | Apache-2.0；`OpenDriveVLA-0.5B` 为 gated、未下载 |
| Source entry | `opendrivevla/modeling_opendrivevla.py`（待 gated source 复核） |
| 当前证据 | L0 repository contract + structural L1 |
| Adapter | 复用 HybridExternalFeature 180 LOC；专属 adapter pending |
| Core op 增量 | 0 |

多相机/BEV/agent/map feature 作为静态 tensor port 或外部 C++ preprocessing
Region 输入；多任务结果作为 named outputs。已编译 bundle 不接受未知动态端口，
扩展输入必须预声明 optional extension port。

不支持项：gated checkpoint/source 完整复核、real capture、artifact/C++ parity。
Memory/performance：pending。
