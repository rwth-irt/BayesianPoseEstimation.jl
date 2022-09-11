# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using CUDA
using MCMCDepth
using Random

# TODO these are experiment specific design decision
parameters = Parameters()

# TODO cpu_model / device_model in main()
# Pose models on CPU since CUDA cannot call OpenGL render functions
t_model = BroadcastedDistribution(KernelNormal, parameters.mean_t, parameters.σ_t)
# TODO sample transformed dist by default?
circular_uniform(::Any) = transformed(KernelCircularUniform())
r_model = BroadcastedDistribution(circular_uniform, Array{parameters.precision}(undef, 3))

# Pixel association on device
uniform(::Any) = transformed(KernelUniform())
o_model = BroadcastedDistribution(uniform, device_array(parameters, parameters.width, parameters.height))

function mix_normal_truncated_exponential(σ::T, θ::T, μ::T, o::T) where {T<:Real}
    # TODO should these generators be part of experiment specific scripts or should I provide some default ones?
    # TODO Compare whether truncated even makes a difference
    dist = KernelBinaryMixture(KernelNormal(μ, σ), truncated(KernelExponential(θ), nothing, μ), o, one(o) - o)
    PixelDistribution(μ, dist)
end
pixel_dist = mix_normal_truncated_exponential | (parameters.pixel_σ, parameters.pixel_θ)

# TODO Do I want to pass pixel_dist as parameter or implement them in a PixelDistributions.jl and eval them from Parameters
# https://bkamins.github.io/julialang/2022/07/15/main.html
"""
    main()
The main inference script which will cause compilation instead of running in the global scope.
"""
# function main(parameters::Parameters, t_model, r_model, o_model, pixel_dist)
# RNGs
dev_rng = device_rng(parameters)
Random.seed!(dev_rng, parameters.seed)
cpu_rng = Random.default_rng()
Random.seed!(cpu_rng, parameters.seed)

# Device
if parameters.device === :CUDA
    CUDA.allowscalar(false)
end
# Allows us to enforce the pose models to run on the CPU
rng_model(rng, model) = RngModel(rng, model)
cpu_model = rng_model | cpu_rng
dev_model = rng_model | dev_rng

# Render context
render_context = RenderContext(parameters.width, parameters.height, parameters.depth, device_array_type(parameters))
scene = Scene(parameters, render_context)

# Assemble PosteriorModel
prior_model = PriorModel(cpu_model(t_model), cpu_model(r_model), dev_model(o_model))
observation_model(μ, o) = ObservationModel(parameters.normalize_img, pixel_dist, μ, o)
posterior_model = PosteriorModel(prior_model, observation_model, render_context, scene, parameters.object_id, parameters.rotation_type)
rand(dev_rng, posterior_model)

# Proposals
# TODO Transformed distributions for SymmetricProposal
# TODO implement getter and test which eval the model symbols


# Finally free OpenGL resources
# SciGL.destroy_render_context(render_context.window)
# end

s = main(parameters, t_model, r_model, o_model, pixel_dist)
plot_depth_img(variables(s).μ |> Array)
plot_depth_img(variables(s).z |> Array)

function test(par::Parameters)
    # WARN not type stable when conditioning on params
    # assoc_is(μ) = KernelNormal(μ, par.pixel_σ)
    # This is since par is evaluated before creating the named function
    # local_σ = par.pixel_σ
    # ass_is(μ) = KernelNormal(μ, local_σ)
    reversed_normal(σ, μ) = KernelNormal(μ, σ)
    assoc_is = reversed_normal | par.pixel_σ
    d = BroadcastedDistribution(assoc_is, 1.0f0)
    logdensityof(d, CUDA.fill(2.0f0, 10))
end

function test_assoc_is(par::Parameters, μ, x)
    # Not type stable
    fn = par.association_is
    # Type stable because of function barrier
    logdensityof(fn(μ), x)
end