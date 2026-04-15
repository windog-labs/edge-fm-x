ARG BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
FROM ${BASE_IMAGE}

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        ninja-build \
        python3-pip \
        python3-pytest && \
    python3 -m pip install --upgrade 'pytest>=8.3,<9' && \
    rm -rf /var/lib/apt/lists/*
