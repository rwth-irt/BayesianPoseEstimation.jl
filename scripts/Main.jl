# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using AbstractMCMC
using Accessors
using CUDA
using MCMCDepth
using Random
# TODO remove
using Test

# TODO these are experiment specific design decision
parameters = Parameters()
parameters = @set parameters.device = :CUDA

# RNGs
rng = cpu_rng(parameters)
dev_rng = device_rng(parameters)
# Allows us to enforce the pose models to run on the CPU
cpu_model = RngModel | rng
dev_model = RngModel | dev_rng

# Prior

# Pose models on CPU to be able to call OpenGL
t_model = ProductBroadcastedDistribution(KernelNormal, parameters.mean_t, parameters.σ_t) |> cpu_model

# TODO sample transformed dist by default?
# circular_uniform(::Any) = transformed(KernelCircularUniform())
r_model = ProductBroadcastedDistribution((x) -> KernelCircularUniform(), cpu_array(parameters, 3)) |> cpu_model

# Association on GPU since we render there
o_model = KernelUniform() |> cpu_model

prior_model = PriorModel(t_model, r_model, o_model)

# Likelihood

function mix_normal_truncated_exponential(σ::T, θ::T, μ::T, o::T) where {T<:Real}
    # TODO Compare whether truncated even makes a difference
    # WARN does not work with o ∈ ℝ, expect o ∈ [0,1]
    dist = KernelBinaryMixture(KernelNormal(μ, σ), truncated(KernelExponential(θ), nothing, μ), o, one(o) - o)
    PixelDistribution(μ, dist)
end
pixel_dist = mix_normal_truncated_exponential | (parameters.pixel_σ, parameters.pixel_θ)

# Proposals
# TODO Transformed distributions for SymmetricProposal
proposal_model = IndependentProposal(prior_model)

# TODO Do I want to pass pixel_dist as parameter or implement them in a PixelDistributions.jl and eval them from Parameters
# https://bkamins.github.io/julialang/2022/07/15/main.html
"""
    main()
The main inference script which will cause compilation instead of running in the global scope.
"""
# function main(parameters::Parameters, t_model, r_model, o_model, pixel_dist)

# Device
if parameters.device === :CUDA
    CUDA.allowscalar(false)
end

# Render context
render_context = RenderContext(parameters.width, parameters.height, parameters.depth, device_array_type(parameters))
scene = Scene(parameters, render_context)
render_propsal = RenderProposal | (parameters.rotation_type, render_context, scene, parameters.object_id)

# Fake observation
observation = render(render_context, scene, parameters.object_id, to_pose(parameters.mean_t, [0, 0, 0])) |> copy

# Assemble PosteriorModel
# WARN use manipulated function since it forces evaluation of parameters to make it type stable
observation_model = ObservationModel | (parameters.normalize_img, pixel_dist)

# TODO transformed if using SymmetricProposal
posterior_model = PosteriorModel(prior_model, observation_model, render_context, scene, parameters.object_id, parameters.rotation_type)
conditioned_posterior = ConditionedModel((; z=observation), posterior_model)

s = @inferred rand(dev_rng, posterior_model)
s2 = @inferred rand(dev_rng, posterior_model)
ℓ = @inferred logdensityof(posterior_model, s)

# true, false
@test s.variables.μ == s2.variables.μ
@test s.variables.t != s2.variables.t

# Sampling algorithm
mh = MetropolisHastings(render_propsal(prior_model), render_propsal(proposal_model))
s1, _ = @inferred AbstractMCMC.step(rng, conditioned_posterior, mh)
# WARN random acceptance needs to be calculated on CPU, thus  CPU rng
s2, _ = @inferred AbstractMCMC.step(rng, conditioned_posterior, mh, s1)

chain = sample(rng, conditioned_posterior, mh, 50000);

# TODO separate evaluation from experiments, i.e. save & load
using Plots
plotly()
density_variable(chain, :t, 20)
density_variable(chain, :r, 20)
polar_density_variable(chain, :r, 20)
