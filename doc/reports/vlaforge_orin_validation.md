# VLAForge Jetson Orin Validation

Date: 2026-07-25

Status: standalone VLAForge runtime and generated Session portability passed in
an emulated JetPack arm64 container. Real Orin GPU execution, latency, power,
and closed-loop behavior remain pending.

## Architecture boundary

VLAForge v0.2 is built from `vlaforge/CMakeLists.txt`. It is intentionally
isolated from the top-level EdgeFM engine/model/operator tree:

- VLAForge core owns Invocation IR, Plan, Session ABI, state/cache/transaction
  semantics, generated C++, and Region plugin contracts.
- EdgeFM can be selected as one TensorRegion artifact provider, but it is not a
  build dependency of the core.
- A model bundle compiles only the Region backends it actually selects.

Consequently, compiling every legacy EdgeFM LLM CUDA operator is not a
VLAForge Orin portability gate. An initial top-level build attempt configured
successfully for Orin and compiled through Ninja step 18/91, but it was stopped
after this scope mismatch was identified. The partial build directory was
removed. This was not a build failure and is not counted as VLAForge evidence.

## Evidence levels

This report keeps four levels separate:

1. x86 host Semantic IR/Plan/generated-C++ correctness;
2. emulated arm64 JetPack compile and CPU execution portability;
3. model-specific arm64 CUDA/TensorRT/AOT artifact execution;
4. real Orin latency, memory, power, and closed-loop behavior.

This run establishes level 2 only. It does not imply that an x86 `.pt2`,
TorchScript, CUDA, or TensorRT artifact ran on Orin.

## Frozen environment

```text
Image: nvcr.io/nvidia/l4t-jetpack:r36.4.0
Image ID:
  sha256:34ccf0f3b63c6da9eee45f2e79de9bf7fdf3beda9abfd72bbf285ae9d40bb673
Image architecture: arm64
Requested platform: linux/arm64
Container compiler: GCC/G++ 11.4.0
CMake: 3.22.1
Ninja: 1.10.1
Build type: Release
Build jobs: 1
```

## arm64 binfmt

The user authorized host arm64 binfmt registration. The active registration is:

```text
/proc/sys/fs/binfmt_misc/qemu-aarch64: enabled
interpreter: /usr/bin/qemu-aarch64
flags: POCF
Docker BuildKit platforms: linux/arm64 available
```

Probe:

```bash
docker run --rm --platform linux/arm64 \
  nvcr.io/nvidia/l4t-jetpack:r36.4.0 \
  sh -lc 'uname -m; dpkg --print-architecture'
```

Observed:

```text
aarch64
arm64
```

## Standalone runtime build

The clean build used the VLAForge source root directly:

```bash
docker run --rm --platform linux/arm64 \
  -v "$PWD":/workspace/edge-fm \
  -w /workspace/edge-fm \
  nvcr.io/nvidia/l4t-jetpack:r36.4.0 \
  bash -lc '
    set -euo pipefail
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y cmake ninja-build
    cmake -S vlaforge -B vlaforge/build-orin -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=ON \
      -DCMAKE_INSTALL_PREFIX=/workspace/edge-fm/vlaforge/build-orin/install
    cmake --build vlaforge/build-orin --parallel 1
    ctest --test-dir vlaforge/build-orin --output-on-failure
    cmake --install vlaforge/build-orin
  '
```

Observed:

```text
build: 18/18
CTest: 6/6 passed
failed: 0
runtime_state_smoke ELF machine: AArch64
```

The six executed tests cover:

1. generic Region C ABI;
2. Tensor/Scalar Region value C ABI;
3. Region C++ ABI;
4. static arena;
5. versioned state and transactional output;
6. generic Session C ABI.

The install tree contains:

```text
lib/libvlaforge_runtime.a
include/vlaforge/runtime/*.h
include/vlaforge/backends/*.h
lib/cmake/VLAForgeRuntime/VLAForgeRuntimeTargets.cmake
lib/cmake/VLAForgeRuntime/VLAForgeRuntimeTargets-release.cmake
```

## Generated no-Python Sessions

The host generated two deterministic v0.2 Sessions:

```text
OpenVLA-like source digest:
  063502957d800a10c19350d1a093d994bfa516908282bb216d384232b2eeca4a
SmolVLA-like source digest:
  6879c2cb2793ff4e904e5a4e30ff2c28842c9788d146c2a4dea8b7901573dc93
```

Each generated source tree was independently configured and compiled inside
the same JetPack arm64 image:

```text
OpenVLA-like: 10/10 build, runner passed
SmolVLA-like: 10/10 build, runner passed
runner ELF machine: AArch64
Python shared-library dependency: absent
```

Both runners were launched with deliberately invalid `PYTHONHOME` and
`PYTHONPATH`. Typed `ModelSession` outputs and generic C ABI outputs were
identical for every Run. The SmolVLA-like runner also exercised queue/cursor
state across Runs and episode reset through the current Adapter template.

These are `fixture-L4` portability results, not real-checkpoint L4 evidence.

## Remaining Orin work

When the real bench and target-specific artifacts are available:

1. build the selected model Region artifacts for JetPack/SM87;
2. bind them through the RegionExecutable ABI;
3. run real-checkpoint Session parity on Orin;
4. measure latency, peak memory, cache benefit, power, and long-run stability;
5. perform closed-loop integration only in the external bottom-software stack.

No sensor synchronization, periodic scheduler, topic subscription, or action
publication is part of this runtime.
