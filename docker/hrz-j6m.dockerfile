ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

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
        python3-pip \
        python3-pytest && \
    rm -rf /var/lib/apt/lists/*
