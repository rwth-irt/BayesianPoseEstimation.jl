# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

# Run this script from the project root, i.e. mcmc-depth-images folder
# Execute the script in a background tmux session to avoid stopping it when disconnecting SSH
julia_remote_nvidia.sh scripts/mcmc_baseline.jl
julia_remote_nvidia.sh scripts/smc_baseline.jl
