# VLAForge reproducibility manifest

Status: **passed**.

## Frozen baseline

- Baseline revision: `f0fc1be`
- Audited revision: `849a7df52567456930af1eca1f00477345df2fa1`
- Frozen core SHA256: `cc2d1b63e2d6cbcd65935b37d69b5f18fae4d2d177c7026a69c6e78f5c80ae6d`
- Frozen core still matches: yes
- Installed-wheel CUDA target: `sm_86`
- Installed-wheel package import: `/tmp/vlaforge-wheel-artifact-eval-849a7df/venv/lib/python3.13/site-packages/vlaforge/__init__.py`

## Durable evidence in Git

- Formal JSON reports: 11
- Committed raw JSON/CSV/Nsight text summaries: 36
- Extracted reproduction commands: 5
- Large checkpoints, AOTI packages and binary profiler databases: not committed

## External archive roots

| Evidence root | Size | Archive? | Status | Reason |
|---|---:|---|---|---|
| `/tmp/vlaforge-smolvla-l3.hr4TVE` | 2.46 GiB | required | present | real SmolVLA export and AOTI artifact reproduction |
| `/tmp/vlaforge-smolvla-l4.oYi5dQ` | 3.82 GiB | required | present | real SmolVLA generated-session support inputs and artifacts |
| `/tmp/vlaforge-smolvla-l4-aligned-0cf3d12` | 0.76 GiB | required | present | final aligned SmolVLA Compile Bundle |
| `/tmp/vlaforge-diffusiondrive-ckpt` | 0.68 GiB | required | present | pinned DiffusionDrive checkpoint |
| `/tmp/vlaforge-diffusiondrive-l3-clean` | 0.74 GiB | required | present | real DiffusionDrive exports and AOTI artifacts |
| `/tmp/vlaforge-diffusiondrive-l4-clean` | 0.24 GiB | required | present | real DiffusionDrive Compile Bundle and deterministic inputs |
| `/tmp/vlaforge-openvla-l3-capture` | 26.54 GiB | required | present | real OpenVLA source exports and deterministic inputs |
| `/tmp/vlaforge-openvla-l3-artifacts` | 79.17 GiB | required | present | 36 normalized OpenVLA exports and sm_86 AOTI packages |
| `/tmp/vlaforge-nsight-v2` | 0.05 GiB | optional | present | optional timeline reinspection; parsed NSYS/NCU summaries and profile hashes are committed |

Required external roots currently total **114.41 GiB**. They must be archived outside Git to retain byte-for-byte real-model reproduction.
The audit found 1 missing ephemeral references. They remain disclosed in the JSON manifest; committed hashes and summaries are still available, while byte-for-byte reruns require regeneration or the external archive roots above.

## Claim boundary

- This manifest covers Host-CUDA artifact reproducibility.
- It does not contain Orin evidence.
- The installed-wheel smoke uses a synthetic tensor Region and is not real-model support evidence.
- Model kernels remain upstream AOTI/cuDNN/CUTLASS/Triton work.
