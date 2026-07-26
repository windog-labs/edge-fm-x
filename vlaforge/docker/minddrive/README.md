# MindDrive 0.5B frontend environment

This image isolates the pinned upstream MindDrive frontend from the VLAForge
runtime. It contains the upstream PyTorch 2.4.1/CUDA 11.8 software profile and
compiles the upstream MMCV/CUDA extensions for RTX 3060 (`sm_86`). It does not
install CARLA, ROS, Cyber, a sensor runtime, or PID/control code.

Build with the pinned upstream source as the Docker context:

```bash
vlaforge/scripts/minddrive/build_frontend_image.sh \
  /home/zhangzimo/Archives/vlaforge-minddrive-0.5b-20260726/source
```

The script rejects a source tree whose Git revision does not match
`1a4085dab1c20895a0c8d2b67b4f8e65712fa8de`. The image is a real-model
frontend/capture environment; generated L4 bundles remain standalone C++ and
do not depend on this Python image.

Docker builds do not expose a GPU to `torch.cuda.is_available()`. The checked-in
`force_cuda_build.patch` changes only the two upstream extension-selection
guards to honor `FORCE_CUDA=1`; the source revision check still runs against
the uncommitted-overlay worktree, and `TORCH_CUDA_ARCH_LIST=8.6` fixes the
generated extension target. The same overlay lazily imports the CARLA-only
runner dependency so offline model construction does not require a simulator,
and adapts the unused fifth `unpad_input` return to flash-attn 2.6.x's
four-value API. These compatibility changes do not alter model math and are
part of the reproducibility contract rather than unrecorded upstream edits.

The base image uses CUDA 11.8 because that is MindDrive's published PyTorch
profile. The host's CUDA 12.8 driver can execute it through NVIDIA Container
Toolkit. AOTInductor artifacts for VLAForge are generated separately for the
host `sm_86` toolchain after frontend capture.
