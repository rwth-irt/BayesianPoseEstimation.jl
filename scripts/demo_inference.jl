# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

using AbstractMCMC: step
using Accessors
using CUDA
using MCMCDepth
using Random
using Plots
using ProgressLogging
using SciGL

CUDA.allowscalar(false)
gr()
MCMCDepth.diss_defaults()

function mtm_parameters()
    parameters = Parameters()
    # NOTE optimal parameter values of pixel_σ and normalization_constant seem to be inversely correlated. Moreover, different values seem to be optimal when using analytic association
    @reset parameters.normalization_constant = 25
    # NOTE Should be able to increase σ in MTM
    @reset parameters.proposal_σ_r = fill(0.2, 3)
    # TODO same seed for experiments
    @reset parameters.seed = rand(RandomDevice(), UInt32)
    @reset parameters.n_steps = 1_000
    @reset parameters.n_burn_in = 0
    @reset parameters.n_thinning = 1
    @reset parameters.n_particles = 50
end

function smc_parameters()
    parameters = Parameters()
    # NOTE SMC: tempering is essential. More steps (MCMC) allows higher normalization_constant than more particles (FP, Bootstrap), 15-30 seems to be a good range
    @reset parameters.normalization_constant = 25
    # TODO same seed for experiments
    @reset parameters.seed = rand(RandomDevice(), UInt32)
    # NOTE resampling dominated like FP & Bootstrap kernels typically perform better with more samples (1_000,100) while MCMC kernels tend to perform better with more steps (2_000,50)
    # TODO Is it really that good? Why all the sudden? Why is MTM so much worse?
    @reset parameters.n_steps = 100
    @reset parameters.n_particles = 50
    # Normalization and tempering leads to less resampling, especially in MCMC sampler
    @reset parameters.relative_ess = 0.5
    # TODO tempering in MCMC?
end

function mh_parameters()
    parameters = Parameters()
    @reset parameters.normalization_constant = 25
    # TODO same seed for experiments
    @reset parameters.seed = rand(RandomDevice(), UInt32)
    @reset parameters.n_steps = 10_000
    # NOTE burn in not required/even harmful if maximum likelihood/posteriori is the goal
    @reset parameters.n_burn_in = 0
    @reset parameters.n_thinning = 1
end

parameters = smc_parameters()

# NOTE takes minutes instead of seconds
# @reset parameters.device = :CPU
cpu_rng = Random.default_rng(parameters)
dev_rng = device_rng(parameters)
gl_context = render_context(parameters)

s_df = scene_dataframe("lm", "test", 2)
row = s_df[101, :]

# Experiment setup
camera = crop_camera(row)
mesh = upload_mesh(gl_context, row.mesh)
@reset mesh.pose = to_pose(row.cam_t_m2c, row.cam_R_m2c)
# Observation is cropped and resized to match the gl_context and crop_camera
mask_img = load_mask_image(row, parameters)
# TODO Add to Parameters. Quite strong prior is required. However, too strong priors are also bad, since the tail distribution would be neglected.
prior_o = mask_img .* 0.6f0 .+ 0.2f0 .|> parameters.float_type |> device_array_type(parameters)
# NOTE Result / conclusion: adding masks makes the algorithm more robust and allows higher σ_t (quantitative difference of how much offset in the prior_t is possible?)
fill!(prior_o, 0.5)

depth_img = load_depth_image(row, parameters) |> device_array_type(parameters)
experiment = Experiment(gl_context, Scene(camera, [mesh]), prior_o, row.cam_t_m2c, depth_img)

# Draw result for visual validation
color_img = load_color_image(row, parameters)
scene = Scene(camera, [mesh])
plot_scene_ontop(gl_context, scene, color_img)

# Model
prior = point_prior(parameters, experiment, cpu_rng)
# posterior = association_posterior(parameters, experiment, prior, dev_rng)
# NOTE no association → prior_o has strong influence
posterior = simple_posterior(parameters, experiment, prior, dev_rng)
# BUG julia 1.9 https://github.com/JuliaGPU/GPUCompiler.jl/issues/384
# posterior = smooth_posterior(parameters, experiment, prior, dev_rng)

# Sampler
parameters = smc_parameters()
sampler = smc_mh(cpu_rng, parameters, experiment, posterior)
# sampler = smc_bootstrap(cpu_rng, parameters, posterior)
# sampler = smc_forward(cpu_rng, parameters, posterior)

# NOTE Benchmark results for smc_mh association & simple ≈ 4.28sec, smooth ≈ 4.74sec
# NOTE diverges if σ_t is too large - masking the image helps. A reasonably strong prior_o also helps to robustify the algorithm
# TODO diagnostics: Accepted steps, resampling steps
final_sample, final_state = smc_inference(cpu_rng, posterior, sampler, parameters)

println("Final log-evidence: $(final_state.log_evidence)")
# WARN final_sample does not represent the final distribution. The final_state does since the samples are weighted. However, for selecting the maximum likelihood sample, no resampling is required.
plot_pose_density(final_sample; trim=false, legend=true)
# plot_prob_img(mean_image(final_sample, :o))
plot_best_pose(final_sample, experiment, color_img)

anim = @animate for i ∈ 0:2:360
    scatter_position(final_sample, 100, label="particle number", camera=(i, 25), projection_type=:perspective, legend_position=:topright)
end;
gif(anim, "anim_fps15.gif", fps=20)


# MCMC samplers
# parameters = mh_Kparameters()
# sampler = mh_sampler(cpu_rng, parameters, experiment, posterior)
# sampler = mh_local_sampler(cpu_rng, parameters, posterior)
parameters = mtm_parameters()
sampler = mtm_sampler(cpu_rng, parameters, experiment, posterior)
# sampler = mtm_local_sampler(cpu_rng, parameters, posterior)
# TODO Diagnostics: Acceptance rate / count, log-likelihood for maximum likelihood selection.
chain = sample(cpu_rng, posterior, sampler, parameters.n_steps; discard_initial=parameters.n_burn_in, thinning=parameters.n_thinning);
# NOTE looks like sampling a pole which is probably sampling uniformly and transforming it back to Euler
plot_pose_chain(chain, 50)
# plot_logprob(chain, 50)
# plot_prob_img(mean_image(chain, :o))
plot_best_pose(chain, experiment, color_img)

# Visualize the maximum posterior
# TODO also track likelihood - plot maximum likelihood pose

anim = @animate for i ∈ 0:2:360
    scatter_position(chain; camera=(i, 25), projection_type=:perspective, legend_position=:topright)
end;
gif(anim, "anim_fps15.gif", fps=20)