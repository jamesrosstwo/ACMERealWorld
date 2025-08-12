FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python 3.10
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    wget \
    build-essential \
    unzip \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.8 \
    python3.8-venv \
    python3.8-dev \
    python3.8-distutils \
    && ln -sf /usr/bin/python3.8 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.8 \
    && rm -rf /var/lib/apt/lists/*

# Confirm versions
RUN python3 --version && pip3 --version

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Install wget to fetch Miniconda
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda on x86 or ARM platforms
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh

RUN git clone git@github.com:jamesrosstwo/ACMERealWorld.git


WORKDIR /ACMERealWorld
RUN git submodule update --init --recursive 
RUN conda env create -f environment.yaml
RUN conda run -n ACMERealWorld pip install -e submodules/gello_software
RUN conda run -n ACMERealWorld pip install -e submoudles/gello_software/third_party/DynamixelSDK/python


ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]

