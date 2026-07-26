# VLAForge Host-CUDA release gate

Status: **passed**.

| Gate | Result |
|---|---:|
| Python | 263 passed, 11 explicit opt-in skipped |
| CUDA AOTI opt-in | 1 passed |
| CPU Release CTest | 7/7 |
| CUDA/AOTI Release CTest | 8/8 |
| CPU installed-package consumer | passed |
| CUDA installed-package consumer | passed |
| Old EdgeFM/custom CUDA sources compiled | no |
| Installed wheel CLI bundle | passed |
| Invalid-Python generated runner | passed |
| Generated runner links libpython | no |

## Installed wheel

- Wheel: `vlaforge-0.2.0.dev0-py3-none-any.whl`
- SHA256: `dd7e176cc351e846787bdb3f0fb7036b32ccf38dc09db48e8c2ee14b5b9deb0a`
- Bundled runtime source entries: 28
- Bundle digest: `36b872bdc340f75f391b423834990a86f39f7236cfc23792da8a73c15e66c352`
- I/O schema digest: `144081eb6287422643a1f88ede09d39652250d16d6442da654bfef9d09e87b5c`
- Provenance: `package:vlaforge-0.2.0.dev0`

## Evidence boundary

- This is an RTX 3060 `sm_86` Host-CUDA release gate.
- It is not Orin latency, power, thermal, or closed-loop evidence.
- Model kernel compilation remains upstream AOTI work.
- VLAForge does not provide sensor synchronization or physical scheduling.
