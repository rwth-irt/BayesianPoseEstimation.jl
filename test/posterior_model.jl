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
rng_model(rng, model) = RngModel(rng, model)
cpu_model = rng_model | rng
curng = CUDA.default_rng()
Random.seed!(curng, 42)

# Parameters
params = MCMCDepth.Parameters()
render_context = RenderContext(params.width, params.height, params.depth, array_for_rng(params.rng))
scene = Scene(params, render_context)
Random.seed!(params.rng, params.seed)
CUDA.allowscalar(false)

# PriorModel
# Pose only makes sense on CPU since CUDA cannot start render calls to OpenGL
mean_t = [0, 0, 1.5] .|> params.precision
σ_t = params.σ_t .|> params.precision
t_model = BroadcastedDistribution(KernelNormal, mean_t, σ_t) |> cpu_model
# TODO This is hacky, any clean implementation which avoids broadcasting over fake parameters?
circular_uniform(::Any) = KernelCircularUniform()
r_model = BroadcastedDistribution(circular_uniform, Array{params.precision}(undef, 3)) |> cpu_model
uniform(::Any) = KernelUniform()
# Use the rng from params
o_model = BroadcastedDistribution(uniform, array_for_rng(params.rng){params.precision}(undef, params.width, params.height))
prior_model = PriorModel(t_model, r_model, o_model)
sample = @inferred rand(params.rng, prior_model)
@test keys(variables(sample)) == (:t, :r, :o)
@test variables(sample).t isa Array{params.precision,1}
@test variables(sample).t |> size == (3,)
@test variables(sample).r isa Array{params.precision,1}
@test variables(sample).r |> size == (3,)
@test variables(sample).o isa array_for_rng(params.rng){params.precision,2}
@test variables(sample).o |> size == (params.width, params.height)
ℓ = @inferred logdensityof(prior_model, sample)
@test ℓ isa Float32

sample5 = @inferred rand(params.rng, prior_model, 5)
@test keys(variables(sample5)) == (:t, :r, :o)
@test variables(sample5).t isa Array{params.precision,2}
@test variables(sample5).t |> size == (3, 5)
@test variables(sample5).r isa Array{params.precision,2}
@test variables(sample5).r |> size == (3, 5)
@test variables(sample5).o isa array_for_rng(params.rng){params.precision,3}
@test variables(sample5).o |> size == (params.width, params.height, 5)
ℓ = @inferred logdensityof(prior_model, sample5)
@test ℓ isa Array{params.precision,1}
@test size(ℓ) == (5,)

# PosteriorModel
function mix_normal_truncated_exponential(σ::T, θ::T, μ::T, o::T) where {T<:Real}
    # TODO should these generators be part of experiment specific scripts or should I provide some default ones?
    # TODO Compare whether truncated even makes a difference
    dist = KernelBinaryMixture(KernelNormal(μ, σ), truncated(KernelExponential(θ), nothing, μ), o, one(o) - o)
    PixelDistribution(μ, dist)
end
my_pixel_dist = mix_normal_truncated_exponential | (0.5f0, 0.5f0)
observation_model(normalize, pixel_dist, μ, o) = ObservationModel(normalize, pixel_dist, μ, o)
obs_model_fn = observation_model | (params.normalize_img, my_pixel_dist)

posterior_model = PosteriorModel(prior_model, obs_model_fn, render_context, scene, params.object_id, params.rotation_type)
sample = @inferred rand(params.rng, posterior_model)
ℓ = @inferred logdensityof(posterior_model, sample)

sample5 = @inferred rand(params.rng, posterior_model, 5)
ℓ5 = @inferred logdensityof(posterior_model, sample5)
