FROM nvidia/cuda:11.5.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir /build

WORKDIR /build

RUN apt-get update && apt-get install -y \
    cmake && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

