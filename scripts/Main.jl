# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using AbstractMCMC
using Accessors
using CUDA
using Distributions
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
r_model = ProductBroadcastedDistribution((_) -> KernelCircularUniform(), cpu_array(parameters, 3)) |> cpu_model
# Scalar prior
o_model = KernelUniform() |> cpu_model
prior_model = PriorModel(t_model, r_model, o_model)

# Likelihood
function mix_normal_truncated_exponential(σ::T, θ::T, μ::T, o::T) where {T<:Real}
    # TODO Compare whether truncated even makes a difference
    dist = KernelBinaryMixture(KernelNormal(μ, σ), truncated(KernelExponential(θ), nothing, μ), o, one(o) - o)
    PixelDistribution(μ, dist)
end
pixel_dist = mix_normal_truncated_exponential | (parameters.pixel_σ, parameters.pixel_θ)

# Proposals
t_proposal = ProductBroadcastedDistribution(KernelNormal, 0, parameters.proposal_σ_t) |> cpu_model
r_proposal = ProductBroadcastedDistribution(KernelNormal, 0, parameters.proposal_σ_r) |> cpu_model
# Scalar prior
o_proposal = KernelNormal(0, parameters.proposal_σ_o) |> cpu_model
# WARN if using Dirac for o, do not propose new o
symmetric_proposal = SymmetricProposal(IndependentModel((; t=t_proposal, r=r_proposal, o=o_proposal)))
independent_proposal = IndependentProposal(prior_model)


# TODO Do I want to pass pixel_dist as parameter in a main function or implement them in a PixelDistributions.jl and eval them from Parameters? Do I even want a main method with a plethora of parameters?
# https://bkamins.github.io/julialang/2022/07/15/main.html

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
posterior_model = PosteriorModel(prior_model, observation_model, render_context, scene, parameters.object_id, parameters.rotation_type)
conditioned_posterior = ConditionedModel((; z=observation), posterior_model)

# Sampling algorithm
mh = MetropolisHastings(render_propsal(prior_model), render_propsal(symmetric_proposal))
s1, _ = @inferred AbstractMCMC.step(rng, conditioned_posterior, mh)
s2, _ = @inferred AbstractMCMC.step(rng, conditioned_posterior, mh, s1)

# WARN random acceptance needs to be calculated on CPU, thus  CPU rng
chain = sample(rng, conditioned_posterior, mh, 10000; discard_initial=250, thinning=2);

# TODO separate evaluation from experiments, i.e. save & load
using Plots
plotly()
model_chain = map(chain) do sample
    s, _ = to_model_domain(sample, mh.bijectors)
    s
end
density_variable(model_chain, :t, 20)
density_variable(model_chain, :r, 20)
# TEST should be approximately n_pixels_rendered / n_pixels
density_variable(model_chain, :o, 20)
polar_histogram_variable(model_chain, :r, 20)
