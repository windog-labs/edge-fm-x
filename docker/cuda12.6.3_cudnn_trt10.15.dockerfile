# syntax=docker/dockerfile:1.4

FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_NO_CACHE_DIR=1 \
    TRT_PACKAGE_DIR=/usr/local/TensorRT \
    LD_LIBRARY_PATH=/usr/local/TensorRT/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib

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

COPY --from=host_trt . /usr/local/TensorRT/
COPY tests/requirements.txt /tmp/edge-fm-tests-requirements.txt
COPY third_party/TensorRT-Edge-LLM/requirements.txt /tmp/trt-edgellm-requirements.txt

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --upgrade 'pytest>=8.3,<9' && \
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.9.1 && \
    grep -vE '^[[:space:]]*torch([<>=~!]|$)' /tmp/edge-fm-tests-requirements.txt > /tmp/edge-fm-tests-runtime.txt && \
    grep -vE '^[[:space:]]*(torch|nvidia-modelopt)(\[.*\])?([<>=~!]|$)' /tmp/trt-edgellm-requirements.txt > /tmp/trt-edgellm-runtime.txt && \
    python3 -m pip install -r /tmp/edge-fm-tests-runtime.txt -r /tmp/trt-edgellm-runtime.txt && \
    rm -f /tmp/edge-fm-tests-requirements.txt /tmp/trt-edgellm-requirements.txt /tmp/edge-fm-tests-runtime.txt /tmp/trt-edgellm-runtime.txt
