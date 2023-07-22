# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using DrWatson
@quickactivate("MCMCDepth")

using Accessors
using BenchmarkTools
using CUDA
using MCMCDepth
using PoseErrors
using Random
using SciGL

CUDA.allowscalar(false)
# Avoid timing the compilation
first_run = true

# Context
parameters = Parameters()
@reset parameters.n_steps = 400
@reset parameters.n_particles = 50

cpu_rng = Random.default_rng(parameters)
dev_rng = device_rng(parameters)
gl_context = render_context(parameters)

bop_subset = ("tless", "test_primesense")
bop_subset_dir = datadir("bop", bop_subset...)
scene_ids = bop_scene_ids(bop_subset_dir)

# TODO run inference and save results on a per-scene basis
scene_id = 1
df = scene_dataframe(bop_subset_dir, scene_ids[scene_id])

# TODO iterate over df rows
# Experiment setup
row = df[100, :]
camera = crop_camera(row)
mesh = upload_mesh(gl_context, load_mesh(row))
@reset mesh.pose = to_pose(row.cam_t_m2c, row.cam_R_m2c)
prior_o = fill(parameters.float_type(parameters.o_mask_not), parameters.width, parameters.height) |> device_array_type(parameters)
# NOTE Result / conclusion: adding masks makes the algorithm more robust and allows higher σ_t (quantitative difference of how much offset in the prior_t is possible?)
mask_img = load_visib_mask_image(row, parameters.img_size...)
prior_o[mask_img] .= parameters.o_mask_is

depth_img = load_depth_image(row, parameters.img_size...) |> device_array_type(parameters)
# TODO do some recall over bias curves?
using LinearAlgebra
pos_bias = 0.0 * normalize(randn(cpu_rng, 3)) .|> parameters.float_type
biased_prior_t = row.cam_t_m2c .+ pos_bias
experiment = Experiment(gl_context, Scene(camera, [mesh]), prior_o, biased_prior_t, depth_img)

# Model
begin
    prior = point_prior(parameters, experiment, cpu_rng)
    # posterior = association_posterior(parameters, experiment, prior, dev_rng)
    # NOTE no association → prior_o has strong influence
    posterior = simple_posterior(parameters, experiment, prior, dev_rng)
    # posterior = smooth_posterior(parameters, experiment, prior, dev_rng)
end
# sampling
begin
    sampler = smc_mh(cpu_rng, parameters, posterior)
    if first_run
        smc_inference(cpu_rng, posterior, sampler, parameters)
        first_run = false
    end
    # TODO tless quite bad without strong prior on mask or position
    runtime = @elapsed begin
        states, final_state = smc_inference(cpu_rng, posterior, sampler, parameters)
    end
end
# BUG for box, smaller side is preferred to be visible - return of the regularization: divide by less -> greater likelihood
begin
    color_img = load_color_image(row, parameters.width, parameters.height)
    plot_best_pose(final_state.sample, experiment, color_img)
end

score, best_idx = findmax(loglikelihood(final_state.sample))

result_root = datadir("exp_raw", bop_subset..., "smc")

@tagsave()
