# VLAForge Host-CUDA release gate

Status: **passed**.

| Gate | Result |
|---|---:|
| Python | 284 passed, 11 explicit opt-in skipped |
| CUDA AOTI opt-in | 1 passed |
| CPU Release CTest | 8/8 |
| CUDA/AOTI Release CTest | 9/9 |
| CPU installed-package consumer | passed |
| CUDA installed-package consumer | passed |
| Old EdgeFM/custom CUDA sources compiled | no |
| Installed wheel CLI bundle | passed |
| Invalid-Python generated runner | passed |
| Generated runner links libpython | no |

## Installed wheel

- Wheel: `vlaforge-0.2.0.dev0-py3-none-any.whl`
- SHA256: `98ce41ba7c49b2ca7ab39bec9b418aaac2dbc254b6593da129c27cc688cb9ff9`
- Bundled runtime source entries: 32
- Bundle digest: `4c9d85f2472d76dca428fd594500a9ce6300b1f432b4ef4492b350475b4771b4`
- I/O schema digest: `144081eb6287422643a1f88ede09d39652250d16d6442da654bfef9d09e87b5c`
- Provenance: `package:vlaforge-0.2.0.dev0`

## Evidence boundary

- This is an RTX 3060 `sm_86` Host-CUDA release gate.
- It is not Orin latency, power, thermal, or closed-loop evidence.
- Model kernel compilation remains upstream AOTI work.
- VLAForge does not provide sensor synchronization or physical scheduling.
