# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using Accessors
using MCMCDepth
using Test

p = Parameters()
p = @set p.precision = Float16
p = @set p.rotation_type = QuatRotation
@test p.precision == Float16

@test p.min_depth isa Float16
@test p.max_depth isa Float16

@test p.pixel_σ isa Float16
@test p.pixel_θ isa Float16
@test p.mix_exponential isa Float16
@test p.static_o |> eltype == Float16

@test p.rotation_type == QuatRotation{Float16}

@test p.mean_t |> eltype == Float16
@test p.σ_t |> eltype == Float16
@test p.proposal_σ_t |> eltype == Float16
@test p.proposal_σ_r |> eltype == Float16
