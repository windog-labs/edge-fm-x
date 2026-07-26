# syntax=docker/dockerfile:1.7

FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive
ARG MINDDRIVE_REVISION=1a4085dab1c20895a0c8d2b67b4f8e65712fa8de

ENV CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    MAX_JOBS=2 \
    TORCH_CUDA_ARCH_LIST=8.6 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        git \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY vlaforge-runtime-requirements.txt /tmp/vlaforge-runtime-requirements.txt
RUN python -m pip install --upgrade "pip<26" "setuptools<76" wheel \
    && python -m pip install -r /tmp/vlaforge-runtime-requirements.txt \
    && python -m pip install --no-build-isolation flash-attn==2.6.3

COPY . /opt/minddrive
RUN git -C /opt/minddrive apply \
        /opt/minddrive/vlaforge-force-cuda-build.patch
RUN test "$(git -C /opt/minddrive rev-parse HEAD)" = "${MINDDRIVE_REVISION}" \
    && python -m pip install \
        --no-build-isolation \
        --no-deps \
        --editable /opt/minddrive

WORKDIR /opt/minddrive

RUN python - <<'PY'
import torch
import transformers

assert torch.__version__.startswith("2.4.1")
assert transformers.__version__ == "4.45.2"
print(
    "minddrive environment ready",
    "torch=" + torch.__version__,
    "cuda=" + str(torch.version.cuda),
    "transformers=" + transformers.__version__,
)
PY
