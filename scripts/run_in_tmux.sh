# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

# Run this script from the project root, i.e. mcmc-depth-images folder
# Execute the script in a background tmux session to avoid stopping it when disconnecting SSH
tmux new-session -s julia -d 'julia_remote_nvidia.sh scripts/smc_bop.jl'
