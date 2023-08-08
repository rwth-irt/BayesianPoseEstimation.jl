# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using DrWatson
@quickactivate("MCMCDepth")

using Accessors
using BenchmarkTools
using CUDA
using CSV
using DataFrames
using MCMCDepth
using PoseErrors
using Random
using SciGL

CUDA.allowscalar(false)
# Avoid timing the compilation
first_run = true

# Context
parameters = Parameters()
@reset parameters.n_steps = 200
@reset parameters.n_particles = 100

cpu_rng = Random.default_rng(parameters)
dev_rng = device_rng(parameters)
gl_context = render_context(parameters)

# Dataset
begin
    bop_subset = ("tless", "test_primesense")
    bop_subset_dir = datadir("bop", bop_subset...)
    scene_ids = bop_scene_ids(bop_subset_dir)
    # TODO run inference and save results on a per-scene basis
    scene_id = 1
    df = gt_targets(bop_subset_dir, scene_ids[scene_id])
    # Experiment setup
    row = df[100, :]
    # TODO incrementally write the dataframe to disk?
    result_df = select(df, :scene_id, :img_id, :obj_id)
    insertcols!(result_df, :est_R => fill(Quaternion(parameters.float_type(1)), nrow(result_df)))
    insertcols!(result_df, :est_t => fill(parameters.float_type[0, 0, 0], nrow(result_df)))
end

# Experiment
begin
    camera = crop_camera(row)
    mesh = upload_mesh(gl_context, load_mesh(row))
    prior_o = fill(parameters.float_type(parameters.o_mask_not), parameters.width, parameters.height) |> device_array_type(parameters)
    # NOTE Result / conclusion: adding masks makes the algorithm more robust and allows higher σ_t (quantitative difference of how much offset in the prior_t is possible?)
    mask_img = load_mask_image(row, parameters.img_size...) |> device_array_type(parameters)
    prior_o[mask_img] .= parameters.o_mask_is

    depth_img = load_depth_image(row, parameters.img_size...) |> device_array_type(parameters)
    prior_t = point_from_segmentation(row.bbox, depth_img, mask_img, row.cv_camera)
    # TODO bias position prior by a fixed distance: recall / bias curve
    # pos_bias = parameters.bias_t * normalize(randn(cpu_rng, 3)) .|> parameters.float_type
    experiment = Experiment(gl_context, Scene(camera, [mesh]), prior_o, prior_t, depth_img)
end

# Model
begin
    prior = point_prior(parameters, experiment, cpu_rng)
    posterior = association_posterior(parameters, experiment, prior, dev_rng)
    # NOTE no association → prior_o has strong influence
    # posterior = simple_posterior(parameters, experiment, prior, dev_rng)
    # posterior = smooth_posterior(parameters, experiment, prior, dev_rng)
end

# sampling
begin
    sampler = smc_mh(cpu_rng, parameters, posterior)
    if first_run
        smc_inference(cpu_rng, posterior, sampler, parameters)
        first_run = false
    end
    runtime = @elapsed begin
        states, final_state = smc_inference(cpu_rng, posterior, sampler, parameters)
    end
end

begin
    color_img = load_color_image(row, parameters.width, parameters.height)
    plot_best_pose(final_state.sample, experiment, color_img)
end

# TODO save DF with estimated pose and distribution?
score, best_idx = findmax(loglikelihood(final_state.sample))

# TODO DrWatson would save each run in a different file and then collect the results in a single DataFrame. However, they would run a simulation per parameter and not per data label.
result_data = Dict("parameters" => parameters, "data" => result_df)
result_root = datadir("exp_raw", bop_subset..., "smc")
@tagsave(datadir(result_root, "sim_1.jld2"), result_data)
# TODO how to read the tags?
