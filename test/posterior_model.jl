# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using CUDA
using Distributions
using MCMCDepth
using Plots
using Random
using SciGL
using Test

const PLOT = true
if PLOT
    pyplot()
end
maybe_plot(fn, x...; y...) = PLOT ? fn(x...; y...) : nothing
rng = Random.default_rng()
Random.seed!(rng, 42)
curng = CUDA.default_rng()


# Parameters
params = MCMCDepth.Parameters()
render_context = RenderContext(params.width, params.height, params.depth, CuArray)
scene = Scene(params, render_context)
Random.seed!(params.rng, params.seed)
CUDA.allowscalar(false)

# PriorModel
mean_t = [0, 0, 1.5] .|> params.precision |> CuArray
σ_t = params.σ_t .|> params.precision |> CuArray
t_model = BroadcastedDistribution(KernelNormal, mean_t, σ_t)
# TODO This is hacky, any clean implementation which avoids broadcasting over fake parameters?
circular_uniform(::Any) = KernelCircularUniform()
r_model = BroadcastedDistribution(circular_uniform, CuVector{params.precision}(undef, 3))
uniform(::Any) = KernelUniform()
o_model = BroadcastedDistribution(uniform, CuMatrix{params.precision}(undef, params.width, params.height))
prior_model = PriorModel(t_model, r_model, o_model)
sample = @inferred rand(params.rng, prior_model)
@test keys(variables(sample)) == (:t, :r, :o)
@test variables(sample).t isa CuArray{params.precision,1}
@test variables(sample).t |> size == (3,)
@test variables(sample).r isa CuArray{params.precision,1}
@test variables(sample).r |> size == (3,)
@test variables(sample).o isa CuArray{params.precision,2}
@test variables(sample).o |> size == (params.width, params.height)
ℓ = @inferred logdensityof(prior_model, sample)
@test ℓ isa Float32

sample5 = @inferred rand(params.rng, prior_model, 5)
@test keys(variables(sample3)) == (:t, :r, :o)
@test variables(sample5).t isa CuArray{params.precision,2}
@test variables(sample5).t |> size == (3, 5)
@test variables(sample5).r isa CuArray{params.precision,2}
@test variables(sample5).r |> size == (3, 5)
@test variables(sample5).o isa CuArray{params.precision,3}
@test variables(sample5).o |> size == (params.width, params.height, 5)
ℓ = @inferred logdensityof(prior_model, sample5)
@test ℓ isa CuArray{params.precision,1}
@test size(ℓ) == (5,)
