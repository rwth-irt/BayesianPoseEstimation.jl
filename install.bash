# Miniconda -> JupyterLab
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
sh /tmp/miniconda.sh -b -p ~/opt/miniconda
rm /tmp/miniconda.sh

source $HOME/opt/miniconda/bin/activate
conda install -c conda-forge -y jupyterlab

# Julia
mkdir -p $HOME/opt
curl -L https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.3-linux-x86_64.tar.gz | tar zx -C $HOME/opt
export PATH="$PATH:$HOME/opt/julia-1.6.3/bin"
julia .devcontainer/install_packages.jl

# IJulia and Revise + config
julia .devcontainer/install_packages.jl
mkdir $HOME/.julia/config/
cp .devcontainer/startup.jl $HOME/.julia/config/startup.jl
