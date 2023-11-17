# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

"""
Used for chains and densities in "Qualitative Analysis of Samplers"
"""

using AbstractMCMC: step
using Accessors
using CUDA
using DataFrames
using MCMCDepth
using Random
using PoseErrors
using SciGL
import CairoMakie as MK

CUDA.allowscalar(false)
diss_defaults()

function mtm_parameters(parameters=Parameters())
    # NOTE optimal parameter values of pixel_σ and c_reg seem to be inversely correlated. Moreover, different values seem to be optimal when using analytic association
    @reset parameters.n_steps = 500
    @reset parameters.n_burn_in = 0
    @reset parameters.n_thinning = 1
    @reset parameters.n_particles = 10
end

function mh_parameters(parameters=Parameters())
    @reset parameters.n_steps = 250
    # NOTE burn in not required/even harmful if maximum likelihood/posteriori is the goal
    @reset parameters.n_burn_in = 0
    @reset parameters.n_thinning = 5
end

function smc_parameters(parameters=Parameters())
    # NOTE SMC: tempering is essential. More steps (MCMC) allows higher c_reg than more particles (FP, Bootstrap)
    # NOTE FP & Bootstrap do not allow independent moves so they profit from a large number of particles. They are also resampling dominated instead of acceptance.
    # NOTE Why is MTM so much worse? One reason might have been that tempering was not implemented.
    @reset parameters.n_steps = 400
    @reset parameters.n_particles = 100
    # Normalization and tempering leads to less resampling, especially in MCMC sampler
    @reset parameters.depth = parameters.n_particles
end

parameters = smc_parameters()
# STERI requires higher resolution for thin instruments
@reset parameters.width = 50;
@reset parameters.height = 50;

# NOTE takes minutes instead of seconds
# @reset parameters.device = :CPU
cpu_rng = Random.default_rng(parameters)
dev_rng = device_rng(parameters)
gl_context = render_context(parameters)

# Vice has distinct features, no occlusions
# df = gt_targets(joinpath("data", "bop", "lm", "test"), 2)
# row = df[101, :]

# Buddha is very smooth without distinct features
# df = gt_targets(joinpath("data", "bop", "lm", "test"), 1)
# row = df[100, :]

# Box shaped object → multimodal for each flat side
# df = gt_targets(joinpath("data", "bop", "tless", "test_primesense"), 1)
# row = df[100, :]

# Clutter and occlusions
# df = gt_targets(joinpath("data", "bop", "tless", "test_primesense"), 18)
# row = df[106, :]

# Parts cut from the image
df = train_targets(joinpath("data", "bop", "itodd", "train_pbr"), 1)
row = df[2, :]

# Small screw
# df = train_targets(joinpath("data", "bop", "itodd", "val"), 1)
# row = df[8, :]

# Steri on flat surface
# df = train_targets(joinpath("data", "bop", "steri", "train_pbr"), 1)
# row = df[5, :]
# # # NOTE high probability for segmentation mask seems beneficial, as well as simple model
# @reset parameters.o_mask_is = 0.95
# @reset parameters.o_mask_not = 1 - parameters.o_mask_is

begin
    # Load Scene
    camera = crop_camera(row)
    mesh = upload_mesh(gl_context, load_mesh(row))
    @reset mesh.pose = to_pose(row.gt_t, row.gt_R)

    # Draw result for visual validation
    color_img = load_color_image(row, parameters.img_size...)
    scene = Scene(camera, [mesh])
    plot_scene_ontop(gl_context, scene, color_img)
end

# Experiment setup
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
# prior_o = parameters.float_type(0.5)
experiment = Experiment(gl_context, Scene(camera, [mesh]), prior_o, prior_t, depth_img)

# Model
prior = point_prior(parameters, experiment, cpu_rng)

posterior = simple_posterior(parameters, experiment, prior, dev_rng)
# posterior = smooth_simple_posterior(parameters, experiment, prior, dev_rng)
# posterior = association_posterior(parameters, experiment, prior, dev_rng)
# posterior = smooth_posterior(parameters, experiment, prior, dev_rng)

# Sampler
parameters = smc_parameters(parameters)
sampler = smc_mh(cpu_rng, parameters, posterior)

# NOTE diverges if σ_t is too large - masking the image helps. A reasonably strong prior_o also helps to robustify the algorithm
@time states, final_state = smc_inference(cpu_rng, posterior, sampler, parameters);

diss_defaults()
# NOTE evidence actually seems to be a pretty good convergence indicator. Once the minimum has been reached, the algorithm seems to have converged.
fig = plot_logevidence(states)
MK.save(joinpath("plots", "evidence_smc_clutter.pdf"), fig)
# Plot state which uses the weights
fig = plot_pose_density(final_state)
MK.save(joinpath("plots", "density_smc_clutter.pdf"), fig)

MK.update_theme!(resolution=(0.5 * DISS_WIDTH, 0.4 * DISS_WIDTH))
fig = plot_best_pose(final_state.sample, experiment, color_img, logprobability)
MK.save(joinpath("plots", "best_smc_clutter.pdf"), fig)
fig = plot_prob_img(mean_image(final_state, :o))
display(fig)
MK.save(joinpath("plots", "prob_img_smc_clutter.pdf"), fig)

# MCMC samplers
parameters = mh_parameters(parameters)
sampler = mh_sampler(cpu_rng, parameters, posterior)
# sampler = mh_local_sampler(cpu_rng, parameters, posterior)
# parameters = mtm_parameters()
# sampler = mtm_sampler(cpu_rng, parameters, posterior);
# sampler = mtm_local_sampler(cpu_rng, parameters, posterior)
# TODO Diagnostics: Acceptance rate / count, log-likelihood for maximum likelihood selection.
@time chain = sample(cpu_rng, posterior, sampler, parameters.n_steps; discard_initial=parameters.n_burn_in, thinning=parameters.n_thinning);

diss_defaults()
fig = plot_pose_chain(chain, 50)
display(fig)
MK.save(joinpath("plots", "density_mcmc_clutter.pdf"), fig)
# plot_logprob(chain, 50)
MK.update_theme!(resolution=(0.5 * DISS_WIDTH, 0.4 * DISS_WIDTH))
# plot_prob_img(mean_image(chain, :o))
fig = plot_best_pose(chain, experiment, color_img)
MK.save(joinpath("plots", "best_mcmc_clutter.pdf"), fig)
diss_defaults()

destroy_context(gl_context)
