# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

# # Mutlistage build for Julia
FROM docker.io/julia:1.11.3-bookworm AS julia-stage

# CUDA
FROM docker.io/nvidia/cuda:12.8.0-runtime-ubuntu22.04

# System packages: dev tools, OpenGL demo
RUN apt-get update && apt-get install -y curl git mesa-utils
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get update && \
    apt-get install -y git-lfs

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # Add sudo support
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Copy Julia from the julia-stage
COPY --from=julia-stage /usr/local/julia /usr/local/julia
# Add Julia binaries to PATH
ENV PATH="/usr/local/julia/bin:${PATH}"

# Set the default user
USER $USERNAME

# Install some common Julia packages
RUN julia -e 'import Pkg; Pkg.add(["ImageShow", "OhMyREPL", "Revise"]); Pkg.instantiate(); Pkg.precompile()'

# Copy startup.jl to Julia config directory
COPY --chown=$USERNAME:$USERNAME .devcontainer/startup.jl /home/$USERNAME/.julia/config/startup.jl
COPY --chown=$USERNAME:$USERNAME *.toml /home/$USERNAME/workspace/

# Setup environment from Manifest.toml
RUN julia --project=/home/$USERNAME/workspace -e 'import Pkg; Pkg.instantiate()'
