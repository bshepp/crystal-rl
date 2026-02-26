# Quantum ESPRESSO + Python RL Environment
# Ubuntu 22.04 with QE 7.3.1 built from source + full Python ML stack

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenmpi-dev \
    openmpi-bin \
    libfftw3-dev \
    libopenblas-dev \
    liblapack-dev \
    wget \
    curl \
    git \
    ca-certificates \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Quantum ESPRESSO 7.3.1 from source
# (apt version is often outdated; building ensures we get the right version)
WORKDIR /opt
RUN wget -q https://github.com/QEF/q-e/archive/refs/tags/qe-7.3.1.tar.gz \
    && tar xzf qe-7.3.1.tar.gz \
    && cd q-e-qe-7.3.1 \
    && ./configure --enable-openmp MPIF90=mpif90 \
    && make -j$(nproc) pw pp bands \
    && mkdir -p /opt/qe-7.3.1/bin \
    && cp bin/* /opt/qe-7.3.1/bin/ \
    && cd / \
    && rm -rf /opt/qe-7.3.1.tar.gz /opt/q-e-qe-7.3.1

ENV PATH="/opt/qe-7.3.1/bin:${PATH}"
ENV ESPRESSO_PSEUDO="/opt/pseudopotentials"

# Download standard pseudopotential library (SSSP Efficiency)
RUN mkdir -p /opt/pseudopotentials

# Python ML/RL stack  - install torch separately with CPU-only index
RUN pip3 install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    ase \
    gymnasium \
    stable-baselines3 \
    tensorboard \
    pandas \
    scikit-learn \
    pymatgen \
    e3nn \
    torch-geometric \
    pyyaml \
    tqdm \
    pytest \
    jarvis-tools \
    mp-api \
    requests

# Working directory
WORKDIR /workspace
COPY . /workspace/

# Default: drop into bash
CMD ["/bin/bash"]
