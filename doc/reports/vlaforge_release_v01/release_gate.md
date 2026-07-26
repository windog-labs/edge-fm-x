# VLAForge Host-CUDA release gate

Status: **passed**.

| Gate | Result |
|---|---:|
| Python | 238 passed, 10 explicit opt-in skipped |
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
- SHA256: `c17d741e6961f28c082f61cfcc23279c5832b1a25def65ad793428623446978a`
- Bundled runtime source entries: 24
- Bundle digest: `325c1c9d95b10a3ecf36a6fc3c3821777170178a1cd9db216eda8c93c9ca4ad0`
- I/O schema digest: `144081eb6287422643a1f88ede09d39652250d16d6442da654bfef9d09e87b5c`
- Provenance: `package:vlaforge-0.2.0.dev0`

## Evidence boundary

- This is an RTX 3060 `sm_86` Host-CUDA release gate.
- It is not Orin latency, power, thermal, or closed-loop evidence.
- Model kernel compilation remains upstream AOTI work.
- VLAForge does not provide sensor synchronization or physical scheduling.
