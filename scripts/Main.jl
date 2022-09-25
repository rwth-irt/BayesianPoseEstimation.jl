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


# TODO Do I want a main method with a plethora of parameters?
# https://bkamins.github.io/julialang/2022/07/15/main.html
# TODO these are experiment specific design decision
parameters = Parameters()
parameters = @set parameters.device = :CUDA

# Device
if parameters.device === :CUDA
    CUDA.allowscalar(false)
end

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
# WARN sampling o with MH seems unstable most of the time
# Scalar prior
o_model = Dirac(parameters.precision(0.5))
o_model = KernelUniform() |> cpu_model
prior_model = PriorModel(t_model, r_model, o_model)

# Proposals
t_proposal = ProductBroadcastedDistribution(KernelNormal, 0, parameters.proposal_σ_t) |> cpu_model
r_proposal = ProductBroadcastedDistribution(KernelNormal, 0, parameters.proposal_σ_r) |> cpu_model
# Scalar prior
o_proposal = KernelNormal(0, parameters.proposal_σ_o)
# WARN if using Dirac for o, do not propose new o
# TODO symmetric_proposal = SymmetricProposal(IndependentModel((; t=t_proposal, r=r_proposal, o=o_proposal)))
symmetric_proposal = SymmetricProposal(IndependentModel((; t=t_proposal, r=r_proposal)))
independent_proposal = IndependentProposal(prior_model)

# Render context
render_context = RenderContext(parameters.width, parameters.height, parameters.depth, device_array_type(parameters))
scene = Scene(parameters, render_context)
render_proposal = RenderModel | (parameters.rotation_type, render_context, scene, parameters.object_id)

# Fake observation
obs_params = @set parameters.mesh_files = ["meshes/monkey.obj", "meshes/cube.obj"]
obs_scene = Scene(obs_params, render_context)
# TODO add scale parameter
obs_scene = @set obs_scene.meshes[2].pose.t = [0.1, 0, 3]
observation = render(render_context, obs_scene, parameters.object_id, to_pose(parameters.mean_t + [0.05, -0.05, -0.1], [0, 0, 0])) |> copy
plot_depth_img(Array(observation))

# TODO wrap in function and move to package
# Calculate normalization constant for the image likelihood
function expected_pixel_count(rng, prior_model, render_context, scene, parameters, n_samples)
    n_pixel = Vector{parameters.precision}(undef, 0)
    for _ in 1:cld(n_samples, parameters.depth)
        prior_sample = rand(rng, prior_model, parameters.depth)
        img = render(render_context, scene, parameters.object_id, to_pose(variables(prior_sample).t, variables(prior_sample).r))
        append!(n_pixel, nonzero_pixels(img, (1, 2)))
    end
    mean(n_pixel)
end
normalization_constant = expected_pixel_count(rng, prior_model, render_context, scene, parameters, 20_000)

# TODO Move to package
# Pixel models
"""
    pixel_mixture(min_depth, max_depth, σ, θ, μ, o)
Mixture distribution for a depth pixel: normal / tail.
The mixture is weighted by the association o for the normal and 1-o for the tail.

* Normal distribution: measuring the object of interest with the expected depth μ and standard deviation σ
* Tail distribution: occlusions (exponential) and random outliers (uniform)
"""
function pixel_mixture(min_depth::T, max_depth::T, σ::T, θ::T, μ::T, o::T) where {T<:Real}
    # TODO Does truncated make a difference?
    # truncate: lower <= upper → max(min_depth, μ)
    exponential = truncated(KernelExponential(θ), min_depth, max(min_depth, μ))
    uniform = KernelUniform(zero(T), max_depth)
    # TODO custom weights for exponential and uniform?
    tail = KernelBinaryMixture(exponential, uniform, one(T), one(T))
    normal = KernelNormal(μ, σ)
    tail = KernelBinaryMixture(exponential, uniform, one(T), one(T))
    KernelBinaryMixture(normal, tail, o, one(o) - o)
end

"""
    pixel_explicit(min_depth, max_depth, σ, θ, μ, o)
Mixture distribution for a depth pixel which explicitly handles invalid μ.
In case the expected depth is invalid, only the tail distribution for outliers is evaluated.
Otherwise, if the measured depth and expected depth are zero, a unreasonably high likelihood would be returned.
The mixture is weighted by the association o for the normal and 1-o for the tail.

* Normal distribution: measuring the object of interest with the expected depth μ and standard deviation σ
* Tail distribution: occlusions (exponential) and random outliers (uniform)
"""
function pixel_explicit(min_depth::T, max_depth::T, σ::T, θ::T, μ::T, o::T) where {T<:Real}
    if μ > 0
        pixel_mixture(min_depth, max_depth, σ, θ, o, μ)
    else
        # TODO weight needed in logdensityof?
        KernelUniform(zero(T), max_depth)
    end
end

# Does not handle invalid μ → ValidPixel & normalization in observation_model
pixel_mix = pixel_mixture | (parameters.min_depth, parameters.max_depth, parameters.pixel_σ, parameters.pixel_θ)
normalized_observation = ObservationModel | (normalization_constant, pixel_mix)
# TEST normalized_posterior actually seems better in a few trials
normalized_posterior = PosteriorModel(prior_model, normalized_observation, render_context, scene, parameters.object_id, parameters.rotation_type)

# Explicitly handles invalid μ → no normalization
pixel_dist = pixel_explicit | (parameters.min_depth, parameters.max_depth, parameters.pixel_σ, parameters.pixel_θ)
explicit_observation = ObservationModel | (pixel_mix)
explicit_posterior = PosteriorModel(prior_model, explicit_observation, render_context, scene, parameters.object_id, parameters.rotation_type)

# Assemble PosteriorModel
# WARN use manipulated function since it forces evaluation of parameters to make it type stable
posterior_sample = rand(dev_rng, normalized_posterior)
plot_depth_img(posterior_sample.variables.z |> Array)

# Sampling algorithm
# conditioned_posterior = ConditionedModel((; z=observation), explicit_posterior)
conditioned_posterior = ConditionedModel((; z=observation), normalized_posterior)
mh = MetropolisHastings(render_proposal(prior_model), render_proposal(symmetric_proposal))
# mh = MetropolisHastings(render_propsal(prior_model), render_propsal(independent_proposal))

# TODO random walk takes longer to converge to correct orientation
# WARN random acceptance needs to be calculated on CPU, thus  CPU rng
# WARN Bad initial sample diverges
chain = sample(rng, conditioned_posterior, mh, 20000; discard_initial=0, thinning=8);

# TODO separate evaluation from experiments, i.e. save & load
using Plots
plotly()
model_chain = map(chain) do sample
    s, _ = to_model_domain(sample, mh.bijectors)
    s
end
plot_variable(model_chain, :t, 100)
plot_variable(model_chain, :r, 100)
# plot_variable(model_chain, :o, 100)
plot_logprob(model_chain, 100)
# density_variable(model_chain, :t, 20)
# density_variable(model_chain, :r, 20)
# # TEST should be approximately n_pixels_rendered / n_pixels
# density_variable(model_chain, :o, 20)
# polar_histogram_variable(model_chain, :r, 20)

# mean(getproperty.(variables.(model_chain), (:t)))
# plot_depth_img(render(render_context, scene, parameters.object_id, to_pose(model_chain[end].variables.t, model_chain[end].variables.r)) |> Array)
