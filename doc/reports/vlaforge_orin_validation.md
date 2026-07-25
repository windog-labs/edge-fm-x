# VLAForge Jetson Orin Validation

Date: 2026-07-25

Status: arm64 container execution ready; full JetPack build and real Orin
hardware execution deferred by user while the test bench is prepared.

## Scope

This report separates three evidence levels:

1. x86 host Invocation IR/C++ correctness;
2. emulated arm64 JetPack compile portability;
3. real Orin GPU execution, latency, power, and closed-loop behavior.

Only the second level can run without an Orin bench. It never implies that an
x86 AOTI/TorchScript artifact ran on Orin.

## Frozen environment

```text
Image: nvcr.io/nvidia/l4t-jetpack:r36.4.0
Image ID:
  sha256:34ccf0f3b63c6da9eee45f2e79de9bf7fdf3beda9abfd72bbf285ae9d40bb673
Image architecture: arm64
Requested platform: linux/arm64
Requested Edge-FM platform: orin
Requested CUDA architecture: SM87
Build jobs: 1
```

## arm64 binfmt

The user authorized host arm64 binfmt registration. Current host state:

```text
/proc/sys/fs/binfmt_misc/qemu-aarch64: present
Docker BuildKit platforms: linux/arm64 available
```

Verification:

```bash
docker run --rm --platform linux/arm64 \
  nvcr.io/nvidia/l4t-jetpack:r36.4.0 uname -m
```

Observed:

```text
aarch64
```

The earlier `exec format error` blocker is resolved.

## Deferred checks

Not run in this v0.2 architecture round:

- top-level EdgeFM `-DPLATFORM=orin` clean configure/build;
- VLAForge runtime arm64 CTest/install/export in JetPack;
- arm64 Region artifact build;
- real Orin CUDA execution;
- model latency, memory, power, and closed-loop tests.

These checks are intentionally deferred so missing bench availability does not
block the host Invocation IR, C++ ABI, model-fixture, and source-audit work.

## Next commands

When the environment is ready:

```bash
docker run --rm --platform linux/arm64 \
  -v "$PWD":/workspace/edge-fm \
  -w /workspace/edge-fm \
  nvcr.io/nvidia/l4t-jetpack:r36.4.0 \
  bash -lc '
    set -euo pipefail
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y cmake ninja-build
    cmake -S . -B build-orin -G Ninja \
      -DPLATFORM=orin \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_PYTHON=OFF
    cmake --build build-orin --parallel 1
  '
```

Then perform a separate VLAForge arm64 clean build/CTest/install, followed by
real Orin model execution with artifacts compiled for the JetPack/SM87
environment.

