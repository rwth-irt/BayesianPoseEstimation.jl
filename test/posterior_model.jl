# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using Accessors
using CUDA
using Distributions
using MCMCDepth
using Plots
using Random
using SciGL
using Test

parameters = Parameters()
parameters = @set parameters.mesh_files = ["meshes/BM067R.obj"]
parameters = @set parameters.device = :CUDA

const PLOT = true
if PLOT
    pyplot()
end
maybe_plot(fn, x...; y...) = PLOT ? fn(x...; y...) : nothing
rng = Random.default_rng()
Random.seed!(rng, 42)
rng_model(rng, model) = RngModel(rng, model)
cpu_model = rng_model | rng
dev_rng = device_rng(parameters)
Random.seed!(dev_rng, 42)

# Parameters
parameters = MCMCDepth.Parameters()
render_context = RenderContext(parameters.width, parameters.height, parameters.depth, device_array_type(parameters))
scene = Scene(parameters, render_context)
Random.seed!(dev_rng, parameters.seed)
CUDA.allowscalar(false)

# PriorModel
# Pose only makes sense on CPU since CUDA cannot start render calls to OpenGL
parameters = @set parameters.mean_t = [0, 0, 1.5]
t_model = ProductBroadcastedDistribution(KernelNormal, parameters.mean_t, parameters.σ_t) |> cpu_model
circular_uniform(::Any) = KernelCircularUniform()
r_model = ProductBroadcastedDistribution(circular_uniform, Array{parameters.precision}(undef, 3)) |> cpu_model
uniform(::Any) = KernelUniform()
# Use the rng from params
o_model = ProductBroadcastedDistribution(uniform, device_array(parameters, parameters.width, parameters.height))
prior_model = PriorModel(t_model, r_model, o_model)
prior_sample = @inferred rand(dev_rng, prior_model)
@test keys(variables(prior_sample)) == (:t, :r, :o)
@test variables(prior_sample).t isa Array{parameters.precision,1}
@test variables(prior_sample).t |> size == (3,)
@test variables(prior_sample).r isa Array{parameters.precision,1}
@test variables(prior_sample).r |> size == (3,)
@test variables(prior_sample).o isa device_array_type(parameters)
@test variables(prior_sample).o |> size == (parameters.width, parameters.height)
@test variables(prior_sample).o |> size == (parameters.width, parameters.height)
ℓ = @inferred logdensityof(prior_model, prior_sample)
@test ℓ isa Float32

sample5 = @inferred rand(dev_rng, prior_model, 5)
@test keys(variables(sample5)) == (:t, :r, :o)
@test variables(sample5).t isa Array{parameters.precision,2}
@test variables(sample5).t |> size == (3, 5)
@test variables(sample5).r isa Array{parameters.precision,2}
@test variables(sample5).r |> size == (3, 5)
@test variables(sample5).o isa device_array_type(parameters)
@test variables(sample5).o |> size == (parameters.width, parameters.height, 5)
ℓ = @inferred logdensityof(prior_model, sample5)
@test ℓ isa Array{parameters.precision,1}
@test size(ℓ) == (5,)

# PosteriorModel
function mix_normal_truncated_exponential(σ::T, θ::T, μ::T, o::T) where {T<:Real}
    # TODO should these generators be part of experiment specific scripts or should I provide some default ones?
    # TODO Compare whether truncated even makes a difference
    dist = KernelBinaryMixture(KernelNormal(μ, σ), truncated(KernelExponential(θ), nothing, μ), o, one(o) - o)
    ValidPixel(μ, dist)
end
my_pixel_dist = mix_normal_truncated_exponential | (0.5f0, 0.5f0)
obs_model_fn = ObservationModel | (10f0, my_pixel_dist)

posterior_model = PosteriorModel(prior_model, obs_model_fn, render_context, scene, parameters.object_id)
posterior_sample = @inferred rand(dev_rng, posterior_model)
ℓ = @inferred logdensityof(posterior_model, posterior_sample)
maybe_plot(plot_depth_img, variables(posterior_sample).μ |> Array)
maybe_plot(plot_depth_img, variables(posterior_sample).z |> Array)

sample5 = @inferred rand(dev_rng, posterior_model, 5)
ℓ5 = @inferred logdensityof(posterior_model, sample5)
