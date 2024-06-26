# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

# Run this script from the project root, i.e. mcmc-depth-images folder
# Execute the script in a background tmux session to avoid stopping it when disconnecting SSH

julia_remote_nvidia.sh --startup-file=no scripts/inference_time.jl
julia_remote_nvidia.sh --startup-file=no scripts/smc_mh_resolution.jl
julia_remote_nvidia.sh --startup-file=no scripts/smc_benchmark.jl
julia_remote_nvidia.sh --startup-file=no scripts/mcmc_benchmark.jl

julia_remote_nvidia.sh --startup-file=no scripts/smc_priors.jl
julia_remote_nvidia.sh --startup-file=no scripts/smc_observation.jl

julia_remote_nvidia.sh --startup-file=no scripts/mcmc_baseline.jl
julia_remote_nvidia.sh --startup-file=no scripts/smc_mh_baseline.jl

julia_remote_nvidia.sh --startup-file=no scripts/smc_bop_val.jl
julia_remote_nvidia.sh --startup-file=no scripts/smc_mh_hyperopt.jl
julia_remote_nvidia.sh --startup-file=no scripts/mcmc_mh_hyperopt.jl
