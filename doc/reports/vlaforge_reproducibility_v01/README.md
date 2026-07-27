# VLAForge reproducibility manifest

Status: **passed**.

## Frozen baseline

- Baseline revision: `f0fc1be`
- Audited revision: `3ce480ef643d0d83a77c9ff5fc8b456f4736bc86`
- Frozen core SHA256: `cc2d1b63e2d6cbcd65935b37d69b5f18fae4d2d177c7026a69c6e78f5c80ae6d`
- Frozen core still matches: yes
- Installed-wheel CUDA target: `sm_86`
- Installed-wheel package import: `/home/zhangzimo/Archives/vlaforge-wheel-3ce480e.fVg5t6/venv/lib/python3.13/site-packages/vlaforge/__init__.py`

## Durable evidence in Git

- Formal JSON reports: 19
- Committed raw JSON/CSV/Nsight text summaries: 316
- Paper, Model Card and figure artifacts: 11
- Extracted reproduction commands: 9
- Large checkpoints, AOTI packages and binary profiler databases: not committed

## External archive roots

| Evidence root | Size | Archive? | Status | Reason |
|---|---:|---|---|---|
| `/home/zhangzimo/Archives/vlaforge-paper-artifact-20260726/required/vlaforge-smolvla-l3.hr4TVE` | 2.46 GiB | required | present | real SmolVLA export and AOTI artifact reproduction |
| `/home/zhangzimo/Archives/vlaforge-paper-artifact-20260726/required/vlaforge-smolvla-l4.oYi5dQ` | 3.82 GiB | required | present | real SmolVLA generated-session support inputs and artifacts |
| `/home/zhangzimo/Archives/vlaforge-paper-artifact-20260726/required/vlaforge-smolvla-l4-aligned-0cf3d12` | 0.76 GiB | required | present | final aligned SmolVLA Compile Bundle |
| `/home/zhangzimo/Archives/vlaforge-paper-artifact-20260726/required/vlaforge-diffusiondrive-ckpt` | 0.68 GiB | required | present | pinned DiffusionDrive checkpoint |
| `/home/zhangzimo/Archives/vlaforge-paper-artifact-20260726/required/vlaforge-diffusiondrive-l3-clean` | 0.74 GiB | required | present | real DiffusionDrive exports and AOTI artifacts |
| `/home/zhangzimo/Archives/vlaforge-paper-artifact-20260726/required/vlaforge-diffusiondrive-l4-clean` | 0.24 GiB | required | present | real DiffusionDrive Compile Bundle and deterministic inputs |
| `/home/zhangzimo/Archives/vlaforge-paper-artifact-20260726/required/vlaforge-openvla-l3-capture` | 26.54 GiB | required | present | real OpenVLA source exports and deterministic inputs |
| `/home/zhangzimo/Archives/vlaforge-paper-artifact-20260726/required/vlaforge-openvla-l3-artifacts` | 79.17 GiB | required | present | 36 normalized OpenVLA exports and sm_86 AOTI packages |
| `/home/zhangzimo/Archives/vlaforge-openvla-l4-stable-5ad7876` | 26.32 GiB | required | present | real OpenVLA weight-paged verified bundle, deterministic inputs, generated no-Python C++ runner, and L4 report |
| `/home/zhangzimo/Archives/vlaforge-paper-artifact-20260726/required/vlaforge-autovla-checkpoint-a7d7ba3` | 15.17 GiB | required | present | pinned 16.29 GB AutoVLA PDMS 89 checkpoint |
| `/home/zhangzimo/Archives/vlaforge-paper-artifact-20260726/required/vlaforge-autovla-l2-f750e54` | 0.13 GiB | required | present | real AutoVLA partitioned L2 exports and capture evidence |
| `/home/zhangzimo/Archives/vlaforge-paper-artifact-20260726/optional/vlaforge-autovla-l3-conservative-f750e54` | 0.14 GiB | optional | present | optional conservative AOTI L3-candidate artifacts |
| `/home/zhangzimo/Archives/vlaforge-minddrive-0.5b-20260726` | 84.09 GiB | required | present | complete MindDrive checkpoint, frontend captures, AOTI artifacts, verified bundle, and generated L4 evidence |
| `/home/zhangzimo/Archives/vlaforge-paper-artifact-20260726/optional/vlaforge-nsight-v2` | 0.05 GiB | optional | present | optional timeline reinspection; parsed NSYS/NCU summaries and profile hashes are committed |

Required external roots currently total **240.13 GiB**. They must be archived outside Git to retain byte-for-byte real-model reproduction.
The audit found 1 missing ephemeral references. They remain disclosed in the JSON manifest; committed hashes and summaries are still available, while byte-for-byte reruns require regeneration or the external archive roots above.

## Claim boundary

- This manifest covers Host-CUDA artifact reproducibility.
- It does not contain Orin evidence.
- The installed-wheel smoke uses a synthetic tensor Region and is not real-model support evidence.
- Model kernels remain upstream AOTI/cuDNN/CUTLASS/Triton work.
