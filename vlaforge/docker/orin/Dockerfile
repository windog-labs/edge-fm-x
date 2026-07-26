ARG L4T_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
FROM ${L4T_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        file \
        git \
        jq \
        libssl-dev \
        ninja-build \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/edge-fm

LABEL org.opencontainers.image.title="VLAForge Orin backend toolchain"
LABEL org.opencontainers.image.description="JetPack arm64 TensorRT build image"
