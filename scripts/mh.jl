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

# Render context
render_context = RenderContext(parameters.width, parameters.height, parameters.depth, device_array_type(parameters))
scene = Scene(parameters, render_context)
render_model = RenderModel | (render_context, scene, parameters.object_id)

# Pose models on CPU to be able to call OpenGL
t_model = ProductBroadcastedDistribution(KernelNormal, parameters.mean_t, parameters.σ_t) |> cpu_model
t_proposal = ProductBroadcastedDistribution(KernelNormal, 0, parameters.proposal_σ_t) |> cpu_model

r_model = ProductBroadcastedDistribution(_ -> KernelCircularUniform(), cpu_array(parameters, 3)) |> cpu_model
r_proposal = ProductBroadcastedDistribution(KernelNormal, 0, parameters.proposal_σ_r)

# WARN sampling o with MH seems unstable most of the time
# WARN if using Dirac for o, do not propose new o
o_model = Dirac(parameters.precision(0.6))
# TODO o_model = KernelUniform() |> cpu_model
o_proposal = KernelNormal(0, parameters.proposal_σ_o)

# Assemble models
prior_model = PriorModel(render_context, scene, parameters.object_id, t_model, r_model, o_model)
# TODO symmetric_proposal = SymmetricProposal(IndependentModel((; t=t_proposal, r=r_proposal, o=o_proposal)))
symmetric_proposal = SymmetricProposal(IndependentModel((; t=t_proposal, r=r_proposal)))
independent_proposal = IndependentProposal(prior_model)

# Normalize the img likelihood to the expected number of rendered expected depth pixels.
# normalization_constant = expected_pixel_count(rng, prior_model, render_context, scene, parameters)
# TODO Using the actual number of pixels makes the model overconfident due to the seemingly large amount of data compared to the prior. Make this adaptive or formalize it?
normalization_constant = parameters.precision(10)

# Pixel models
# Does not handle invalid μ → ValidPixel & normalization in observation_model
pixel_mix = pixel_mixture | (parameters.min_depth, parameters.max_depth, parameters.pixel_σ, parameters.pixel_θ)
normalized_observation = ObservationModel | (normalization_constant, pixel_mix)
# TEST normalized_posterior actually seems better in a few trials
normalized_posterior = PosteriorModel(prior_model, normalized_observation)

# Explicitly handles invalid μ → no normalization
pixel_expl = pixel_explicit | (parameters.min_depth, parameters.max_depth, parameters.pixel_σ, parameters.pixel_θ)
explicit_observation = ObservationModel | (pixel_expl)
explicit_posterior = PosteriorModel(prior_model, explicit_observation)

# Assemble PosteriorModel
# WARN use manipulated function since it forces evaluation of parameters to make it type stable
posterior_sample = rand(dev_rng, normalized_posterior)
plot_depth_img(posterior_sample.variables.z |> Array)
posterior_sample = rand(dev_rng, explicit_posterior)
plot_depth_img(posterior_sample.variables.z |> Array)

# Fake observation
# TODO occlusion
obs_params = @set parameters.mesh_files = ["meshes/monkey.obj", "meshes/cube.obj"]
obs_scene = Scene(obs_params, render_context)
obs_scene = @set obs_scene.meshes[2].pose.translation = Translation(0.1, 0, 3)
obs_scene = @set obs_scene.meshes[2].scale = Scale(1.8, 1.5, 1)
obs_μ = render(render_context, obs_scene, parameters.object_id, to_pose(parameters.mean_t + [0.05, -0.05, -0.1], [0, 0, 0]))
obs = rand(dev_rng, explicit_observation(obs_μ, 0.8f0))
plot_depth_img(Array(obs))

# Sampling algorithm
# conditioned_posterior = ConditionedModel((; z=obs), explicit_posterior)
conditioned_posterior = ConditionedModel((; z=obs), normalized_posterior)
mh = MetropolisHastings(prior_model, render_model(symmetric_proposal))
# mh = MetropolisHastings(render_model(prior_model), render_model(independent_proposal))

# TODO random walk takes longer to converge to correct orientation
# WARN random acceptance needs to be calculated on CPU, thus  CPU rng
# WARN Bad initial sample diverges
chain = sample(rng, conditioned_posterior, mh, 20000; discard_initial=0, thinning=3);

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
density_variable(model_chain, :t, 20)
# # TEST should be approximately n_pixels_rendered / n_pixels
density_variable(model_chain, :o, 20)
polar_histogram_variable(model_chain, :r; nbins=20)

# mean(getproperty.(variables.(model_chain), (:t)))
plot_depth_img(render(render_context, scene, parameters.object_id, to_pose(model_chain[end].variables.t, model_chain[end].variables.r)) |> Array)
