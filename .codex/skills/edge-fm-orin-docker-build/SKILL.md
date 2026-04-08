---
name: edge-fm-orin-docker-build
description: Build edge-fm for Jetson Orin inside a mounted Docker workspace using `nvcr.io/nvidia/l4t-jetpack:r36.4.0`. Use when Codex needs to validate Orin compilation, reproduce a JetPack 6.1 build environment, or produce a repeatable Docker-based Orin build command without relying on Thor.
---

# edge-fm Orin Docker Build

这个 skill 只用于 `Orin`，不要把 `thor` 当成替代平台。默认平台固定为 `-DPLATFORM=orin`，镜像固定为 `nvcr.io/nvidia/l4t-jetpack:r36.4.0`。

## 什么时候用

- 用户要验证 edge-fm 能否在 Jetson Orin / JetPack 6.1 环境下编译通过。
- 用户要一个不污染宿主机的挂载式 Docker 构建流程。
- 用户要复现 Orin 的 CUDA / cuDNN / TensorRT 头文件和库路径。

## 工作流

1. 先确认镜像已可用；如果本地没有，再执行：
   ```bash
   docker pull nvcr.io/nvidia/l4t-jetpack:r36.4.0
   ```
2. 使用挂载本地目录的方式进入容器，工作目录固定到仓库根目录：
   ```bash
   export EDGE_FM_BUILD_JOBS="${EDGE_FM_BUILD_JOBS:-1}"
   docker run --rm --platform linux/arm64 \
     -v "$PWD":/workspace/edge-fm \
     -w /workspace/edge-fm \
     nvcr.io/nvidia/l4t-jetpack:r36.4.0 \
     bash -lc 'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y cmake ninja-build && cmake -S . -B build-orin -G Ninja -DPLATFORM=orin -DCMAKE_BUILD_TYPE=Release && cmake --build build-orin --parallel "${EDGE_FM_BUILD_JOBS}"'
   ```
3. 如果需要 Python 绑定安装产物，再补：
   ```bash
   export EDGE_FM_BUILD_JOBS="${EDGE_FM_BUILD_JOBS:-1}"
   docker run --rm --platform linux/arm64 \
     -v "$PWD":/workspace/edge-fm \
     -w /workspace/edge-fm \
     nvcr.io/nvidia/l4t-jetpack:r36.4.0 \
     bash -lc 'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y cmake ninja-build && cmake -S . -B build-orin -G Ninja -DPLATFORM=orin -DCMAKE_BUILD_TYPE=Release && cmake --build build-orin --parallel "${EDGE_FM_BUILD_JOBS}" && cmake --install build-orin'
   ```

## 实施要求

- 只使用挂载目录，不要把源码 `COPY` 进镜像。
- 只构建 `orin`，不要改成 `thor`，也不要用 `thor` 的编译选项代替 `orin`。
- 配置阶段必须显式传 `-DPLATFORM=orin`。
- 默认构建目录使用 `build-orin/`，避免污染通用 `build/`。
- 依赖安装只补容器内缺失的基础工具，至少包含 `cmake` 和 `ninja-build`。
- 默认并发建议从 `EDGE_FM_BUILD_JOBS=1` 起步；Jetson/JetPack 容器内某些 CUDA 模板单元内存开销很大，先保证稳定通过，再按机器内存上调。
- 如果需要排查 CUDA / cuDNN 发现问题，优先检查：
  - `/usr/local/cuda/targets/aarch64-linux/include`
  - `/usr/lib/aarch64-linux-gnu`
  - `/usr/include/aarch64-linux-gnu`

## 验证

1. 先看 `cmake -S . -B build-orin -DPLATFORM=orin` 是否通过。
2. 再看 `cmake --build build-orin --parallel` 是否完成。
3. 若用户要求安装产物，确认 `build-orin/install/` 下出现 `lib/` 与 `python/`。
