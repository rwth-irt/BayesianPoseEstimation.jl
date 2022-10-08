# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

# WARN Do not run this if you want Revise to work
# include("../src/MCMCDepth.jl")
# using .MCMCDepth

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
# TODO lot of copy - paste. Function to generate samplers?
t_model = ProductBroadcastedDistribution(KernelNormal, parameters.mean_t, parameters.σ_t) |> cpu_model
# r_model = ProductBroadcastedDistribution(_ -> KernelCircularUniform(), cpu_array(parameters, 3)) |> cpu_model
r_model = QuaternionDistribution(parameters.precision) |> cpu_model
# TODO Show robustness for different o by estimating it
# o_model = Dirac(parameters.precision(3 / 4)) |> cpu_model
# TODO works but takes longer to converge
o_model = KernelUniform() |> cpu_model
prior_model = PriorModel(render_context, scene, parameters.object_id, t_model, r_model, o_model)

# t & r change expected depth, o not
t_independent = IndependentProposal(IndependentModel((; t=t_model))) |> render_model
t_sym_dist = ProductBroadcastedDistribution(KernelNormal, 0, parameters.proposal_σ_t) |> cpu_model
t_symmetric = SymmetricProposal(IndependentModel((; t=t_sym_dist))) |> render_model

r_independent = IndependentProposal(IndependentModel((; r=r_model))) |> render_model
# r_proposal = ProductBroadcastedDistribution(KernelNormal, 0, parameters.proposal_σ_r) |> cpu_model
# r_symmetric = SymmetricProposal(IndependentModel((; r=r_proposal))) |> render_model
r_sym_dist = r = QuaternionPerturbation(parameters.proposal_σ_r_quat) |> cpu_model
r_symmetric = QuaternionProposal(IndependentModel((; r=r_sym_dist))) |> render_model

o_independent = IndependentProposal(IndependentModel((; o=o_model)))
o_sym_dist = KernelNormal(0, parameters.proposal_σ_o) |> cpu_model
o_symmetric = SymmetricProposal(IndependentModel((; o=o_sym_dist)))

# Assemble samplers
t_ind_mh = MetropolisHastings(prior_model, t_independent)
t_sym_mh = MetropolisHastings(prior_model, t_symmetric)

r_ind_mh = MetropolisHastings(prior_model, r_independent)
r_sym_mh = MetropolisHastings(prior_model, r_symmetric)

o_ind_mh = MetropolisHastings(prior_model, o_independent)
o_sym_mh = MetropolisHastings(prior_model, o_symmetric)

ind_sym_sampler = ComposedSampler(Weights([0.1, 0.1, 0.1, 1.0, 1.0, 1.0]), t_ind_mh, r_ind_mh, o_ind_mh, t_sym_mh, r_sym_mh, o_sym_mh)
ind_sampler = ComposedSampler(t_ind_mh, r_ind_mh, o_ind_mh)
# TODO works better than combination? Maybe because we do not sample the poles like with RotXYZ
sym_sampler = ComposedSampler(t_sym_mh, r_sym_mh, o_sym_mh)

# Normalize the img likelihood to the expected number of rendered expected depth pixels.
# TODO Using the actual number of pixels makes the model overconfident due to the seemingly large amount of data compared to the prior. Make this adaptive or formalize it?
normalization_constant = parameters.precision(10)
# TODO normalization_constant = expected_pixel_count(rng, prior_model, render_context, scene, parameters)

# Pixel models
# Does not handle invalid μ → ValidPixel & normalization in observation_model
pixel_mix = pixel_mixture | (parameters.min_depth, parameters.max_depth, parameters.pixel_σ, parameters.pixel_θ)
normalized_observation = ObservationModel | (normalization_constant, pixel_mix)
normalized_posterior = PosteriorModel(prior_model, normalized_observation)

# Explicitly handles invalid μ → no normalization
pixel_expl = pixel_explicit | (parameters.min_depth, parameters.max_depth, parameters.pixel_σ, parameters.pixel_θ)
explicit_observation = ObservationModel | (pixel_expl)
explicit_posterior = PosteriorModel(prior_model, explicit_observation)

# Fake observation
# TODO occlusion
obs_params = @set parameters.mesh_files = ["meshes/monkey.obj", "meshes/cube.obj"]
obs_scene = Scene(obs_params, render_context)
obs_scene = @set obs_scene.meshes[2].pose.translation = Translation(0.1, 0, 3)
obs_scene = @set obs_scene.meshes[2].scale = Scale(1.8, 1.5, 1)
obs_μ = render(render_context, obs_scene, parameters.object_id, to_pose(parameters.mean_t + [0.05, -0.05, -0.1], [0, 0, 0]))
obs = rand(dev_rng, explicit_observation(obs_μ, 0.8f0))
plot_depth_img(Array(obs))

# PosteriorModel
# TODO normalized_posterior seems way better
# conditioned_posterior = ConditionedModel((; z=obs), explicit_posterior);
conditioned_posterior = ConditionedModel((; z=obs), normalized_posterior)

# WARN random acceptance needs to be calculated on CPU, thus CPU rng
# TODO rng
chain = sample(rng, conditioned_posterior, ind_sym_sampler, 10_000; discard_initial=5_000, thinning=2);

# TODO separate evaluation from experiments, i.e. save & load
using Plots
model_chain = map(chain) do sample
    # TODO function for different samplers
    s, _ = to_model_domain(sample, bijector(prior_model))
    s
end;
plotly()
plot_variable(model_chain, :t, 100)
plot_variable(model_chain, :r, 100)
# plot_variable(model_chain, :o, 100)
plot_logprob(model_chain, 100)
density_variable(model_chain, :t)
density_variable(model_chain, :r)
density_variable(model_chain, :o)
# # pyplot()
# polar_histogram_variable(model_chain, :r; nbins=180)

# # mean(getproperty.(variables.(model_chain), (:t)))
plot_depth_img(render(render_context, scene, parameters.object_id, to_pose(model_chain[end].variables.t, model_chain[end].variables.r)) |> Array)
