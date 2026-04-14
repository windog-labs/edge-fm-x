FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TRT_PACKAGE_DIR=/usr/local/TensorRT

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        cmake \
        git \
        make \
        ninja-build \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-pytest \
        rsync && \
    rm -rf /var/lib/apt/lists/*

# TensorRT is expected at /usr/local/TensorRT. The docker build wrappers will
# mount a host TensorRT installation there when the image itself does not bake
# one in.
RUN mkdir -p /usr/local/TensorRT
