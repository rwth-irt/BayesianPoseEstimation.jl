# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

using AbstractMCMC: step
using Accessors
using CUDA
using MCMCDepth
using Random
using PoseErrors
using SciGL

CUDA.allowscalar(false)
diss_defaults()

function mtm_parameters()
    parameters = Parameters()
    # NOTE optimal parameter values of pixel_σ and c_reg seem to be inversely correlated. Moreover, different values seem to be optimal when using analytic association
    @reset parameters.seed = rand(RandomDevice(), UInt32)
    @reset parameters.n_steps = 500
    @reset parameters.n_burn_in = 0
    @reset parameters.n_thinning = 1
    @reset parameters.n_particles = 10
end

function mh_parameters()
    parameters = Parameters()
    @reset parameters.seed = rand(RandomDevice(), UInt32)
    @reset parameters.n_steps = 250
    # NOTE burn in not required/even harmful if maximum likelihood/posteriori is the goal
    @reset parameters.n_burn_in = 0
    @reset parameters.n_thinning = 5
end

function smc_parameters()
    parameters = Parameters()
    # NOTE SMC: tempering is essential. More steps (MCMC) allows higher c_reg than more particles (FP, Bootstrap)
    @reset parameters.seed = rand(RandomDevice(), UInt32)
    # NOTE FP & Bootstrap do not allow independent moves so they profit from a large number of particles. They are also resampling dominated instead of acceptance.
    # NOTE Why is MTM so much worse? One reason might have been that tempering was not implemented.
    @reset parameters.n_steps = 200
    @reset parameters.n_particles = 100
    # Normalization and tempering leads to less resampling, especially in MCMC sampler
    @reset parameters.depth = parameters.n_particles
end

parameters = smc_parameters()

# NOTE takes minutes instead of seconds
# @reset parameters.device = :CPU
cpu_rng = Random.default_rng(parameters)
dev_rng = device_rng(parameters)
gl_context = render_context(parameters)

# Vice has distinct features, no occlusions
df = gt_targets(joinpath("data", "bop", "lm", "test"), 2)
row = df[101, :]

# Buddha is very smooth without distinct features
# df = gt_targets(joinpath("data", "bop", "lm", "test"), 1)
# row = df[100, :]

# Box shaped object → multimodal for each flat side
# df = gt_targets(joinpath("data", "bop", "tless", "test_primesense"), 1)
# row = df[100, :]

# Clutter and occlusions
# NOTE better crop → better result if using union in ℓ normalization
# df = gt_targets(joinpath("data", "bop", "tless", "test_primesense"), 18)
# row = df[298, :]

# Experiment setup
camera = crop_camera(row)
mesh = upload_mesh(gl_context, load_mesh(row))
@reset mesh.pose = to_pose(row.gt_t, row.gt_R)
# Observation is cropped and resized to match the gl_context and crop_camera
depth_img = load_depth_image(row, parameters.img_size...) |> device_array_type(parameters)
mask_img = load_mask_image(row, parameters.img_size...) |> device_array_type(parameters)
prior_o = fill(parameters.float_type(parameters.o_mask_not), parameters.width, parameters.height) |> device_array_type(parameters)
# NOTE Result / conclusion: adding masks makes the algorithm more robust and allows higher σ_t (quantitative difference of how much offset in the prior_t is possible?)
prior_o[mask_img] .= parameters.o_mask_is
prior_t = point_from_segmentation(row.bbox, depth_img, mask_img, row.cv_camera)
# For RFID scenario
# prior_t = row.gt_t + rand(cpu_rng, KernelNormal(0, 0.01f0), 3)
# prior_o .= 0.5
experiment = Experiment(gl_context, Scene(camera, [mesh]), prior_o, prior_t, depth_img)

# Draw result for visual validation
color_img = load_color_image(row, parameters.img_size...)
scene = Scene(camera, [mesh])
plot_scene_ontop(gl_context, scene, color_img)

# Model
prior = point_prior(parameters, experiment, cpu_rng)

# NOTE no association → prior_o has strong influence
posterior = simple_posterior(parameters, experiment, prior, dev_rng)
# posterior = association_posterior(parameters, experiment, prior, dev_rng)
# posterior = smooth_posterior(parameters, experiment, prior, dev_rng)

# Sampler
parameters = smc_parameters()
sampler = smc_mh(cpu_rng, parameters, posterior)
# sampler = smc_bootstrap(cpu_rng, parameters, posterior)
# sampler = smc_forward(cpu_rng, parameters, posterior)

# NOTE Benchmark results for smc_mh association & simple ≈ 4.28sec, smooth ≈ 4.74sec
# NOTE diverges if σ_t is too large - masking the image helps. A reasonably strong prior_o also helps to robustify the algorithm
# TODO diagnostics: Accepted steps, resampling steps
@time states, final_state = smc_inference(cpu_rng, posterior, sampler, parameters);
# NOTE evidence actually seems to be a pretty good convergence indicator. Once the minimum has been reached, the algorithm seems to have converged.
fig = plot_logevidence(states)
# Plot state which uses the weights
plot_pose_density(final_state.sample)
# plot_prob_img(mean_image(final_sample, :o))
plot_best_pose(final_state.sample, experiment, color_img, logprobability)

# TODO
# step_size = length(states) ÷ 100
# anim = @animate for idx in 1:step_size:length(states)
#     # White background required for accurate axis colors
#     plot_best_pose(states[idx].sample, experiment, color_img; title="Iteration $(idx)", background_color=:white)
# end;
# gif(anim, "smc.gif", fps=15)

# MCMC samplers
# parameters = mh_parameters()
# sampler = mh_sampler(cpu_rng, parameters, posterior)
# sampler = mh_local_sampler(cpu_rng, parameters, posterior)
parameters = mtm_parameters()
sampler = mtm_sampler(cpu_rng, parameters, posterior);
# sampler = mtm_local_sampler(cpu_rng, parameters, posterior)
# TODO Diagnostics: Acceptance rate / count, log-likelihood for maximum likelihood selection.
@time chain = sample(cpu_rng, posterior, sampler, parameters.n_steps; discard_initial=parameters.n_burn_in, thinning=parameters.n_thinning);
# NOTE looks like sampling a pole which is probably sampling uniformly and transforming it back to Euler
plot_pose_chain(chain, 50)
# plot_logprob(chain, 50)
# plot_prob_img(mean_image(chain, :o))
plot_best_pose(chain, experiment, color_img)

# TODO
# step_size = length(chain) ÷ 100
# anim = @animate for idx in 1:step_size:length(chain)
#     # White background required for accurate axis colors
#     plot_best_pose(chain[idx], experiment, color_img; title="Iteration $(idx)", background_color=:white)
# end;
# gif(anim, "mcmc.gif"; fps=15)

destroy_context(gl_context)
