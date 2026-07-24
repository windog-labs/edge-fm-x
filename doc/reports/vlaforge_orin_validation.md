# VLAForge Jetson Orin Validation

Date: 2026-07-24

Status: blocked before container startup by missing host arm64 emulation.

## Scope

This is a separate deployment-environment report. It was started only after
the local real-model paper matrix and release regression passed. It does not
replace the x86 SmolVLA/OpenVLA correctness and performance evidence.

The intended checks are:

1. configure and build the top-level Edge-FM project with
   `-DPLATFORM=orin`;
2. build, test, and install the VLAForge C++17 runtime and fixture backend in
   the same arm64 JetPack environment;
3. keep real backend artifact execution separate, because the current CUDA
   AOTInductor and CPU TorchScript packages were produced for the local x86
   PyTorch environment and are not Orin deployment artifacts.

## Frozen environment

```text
Image: nvcr.io/nvidia/l4t-jetpack:r36.4.0
Image ID:
  sha256:34ccf0f3b63c6da9eee45f2e79de9bf7fdf3beda9abfd72bbf285ae9d40bb673
Image architecture: arm64
Image size: 5,605,246,675 bytes
Image created: 2024-10-24T05:52:42.766632237Z
Host Docker client/server: 29.2.1 / 29.2.1
Requested platform: linux/arm64
Requested Edge-FM platform: orin
Requested CUDA architecture: SM87
Build jobs: 1
```

The image was pulled successfully and remains in the local Docker cache.

## Exact attempted command

```bash
docker run --rm --platform linux/arm64 \
  -v "$PWD":/workspace/edge-fm \
  -w /workspace/edge-fm \
  nvcr.io/nvidia/l4t-jetpack:r36.4.0 \
  bash -lc '
    set -euo pipefail
    echo "container_arch=$(uname -m)"
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y cmake ninja-build
    cmake -S . -B build-orin -G Ninja \
      -DPLATFORM=orin \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_PYTHON=OFF
    cmake --build build-orin --parallel 1
  '
```

Observed result:

```text
exec /usr/bin/bash: exec format error
exit code: 255
```

A minimal `/bin/true` probe fails with the same error. The host has neither a
`qemu-aarch64-static` executable nor a `qemu-aarch64` entry under
`/proc/sys/fs/binfmt_misc`. `build-orin/` was not created, so no configure or
compile result is claimed.

## Required next authority

The normal Docker remedy is a privileged, host-wide binfmt registration:

```bash
docker run --privileged --rm tonistiigi/binfmt --install arm64
```

After explicit authorization, verify the registration and rerun the frozen
command above:

```bash
cat /proc/sys/fs/binfmt_misc/qemu-aarch64
docker run --rm --platform linux/arm64 \
  nvcr.io/nvidia/l4t-jetpack:r36.4.0 uname -m
```

Expected architecture output is `aarch64`. Registration changes host kernel
state and therefore was not performed implicitly.

## Current conclusion

- Orin image availability: passed.
- Image identity and architecture: passed.
- Host arm64 container execution: blocked by missing binfmt.
- Edge-FM `PLATFORM=orin` configure/build: not run.
- VLAForge arm64 runtime build/CTest/install: not run.
- Real SmolVLA/OpenVLA execution on Orin hardware: not run and not implied.

