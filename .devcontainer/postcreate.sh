# Julia
mkdir -p /home/vscode/julia && \
    curl -L https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.3-linux-x86_64.tar.gz | tar zx -C /home/vscode/julia

# JupyterLab
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh &&
    sh /tmp/miniconda.sh -b -p /home/vscode/miniconda &&
    rm /tmp/miniconda.sh
conda init bash
conda install -c conda-forge -y jupyterlab

# IJulia and Revise + config
julia .devcontainer/install_packages.jl
cp .devcontainer/startup.jl /home/vscode/.julia/config/startup.jl

# Activate and precompile current package
julia .devcontainer/postcreate.jl
