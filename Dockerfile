ARG CUDA_VERSION=12.4.0
ARG from=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

FROM ${from} as base

ARG DEBIAN_FRONTEND=noninteractive

# ENV USERNAME=user-name-goes-here
# ENV USER_UID=1000
# ENV USER_GID=$USER_UID

# RUN groupadd --gid $USER_GID $USERNAME \
#     && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
#     #
#     # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
#     && apt-get update \
#     && apt-get install -y sudo \
#     && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
#     && chmod 0440 /etc/sudoers.d/$USERNAME

RUN <<EOF
apt update -y && apt upgrade -y && apt install -y --no-install-recommends  \
    git \
    git-lfs \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    vim \
    libsndfile1 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    redis-server \
&& rm -rf /var/lib/apt/lists/*
EOF

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN git lfs install

ENV CUDA_HOME=/usr/local/cuda
ENV TRANSFORMERS_CACHE=/app/models


WORKDIR /app

FROM base as dev

FROM dev as bundle_req
RUN pip3 install --no-cache-dir networkx==3.1

COPY requirements_fp8.txt /tmp/
RUN pip3 install -U -r /tmp/requirements_fp8.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install torchsde

# RUN pip3 install --no-cache-dir \
#     gradio==4.42.0 \
#     gradio_client==1.3.0 \
#     transformers-stream-generator==0.0.4 \
#     dashscope \
#     numpy \ 
#     gekko \
#     pandas \
#     openai \
#     huggingface_hub \
#     deepspeed

COPY requirements_server_flask.txt /tmp/
RUN pip3 install -U -r /tmp/requirements_server_flask.txt

COPY requirements_server_fastapi.txt /tmp/
RUN pip3 install -U -r /tmp/requirements_server_fastapi.txt

EXPOSE 8080
ENV HF_HUB_CACHE=/app/models
ENV HF_HOME=/app/models
RUN export HF_HOME=/app/models
RUN export HF_HUB_CACHE=/app/models
RUN export TRANSFORMERS_CACHE=/app/models



CMD ["/bin/bash"]