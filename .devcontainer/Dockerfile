# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2020, Tim Redick
# All rights reserved.

# OpenGL
FROM ubuntu:20.04

# dev tools
RUN apt-get update && apt-get install -y curl git
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get update && \
    apt-get install -y git-lfs

# Non-root-user to avoid file permission problems
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    apt-get update && \
    apt-get install -y sudo && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL >/etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME
USER $USERNAME

# Environment setup
ENV PATH="/home/vscode/miniconda/bin:${PATH}"
ENV PATH="/home/vscode/julia/julia-1.6.3/bin:${PATH}"
