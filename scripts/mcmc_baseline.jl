# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
Run different MCMC algorithms on the synthetic BOP datasets:
* Metropolis Hastings
* Multiple Try Metropolis

Model setup for segmentation:
* Segmentation prior for position `t` and pixel association `o`
* Simple likelihood function with mixture model for the pixels, a simple regularization, and without modeling the association,
"""

using DrWatson
@quickactivate("MCMCDepth")

using Accessors
using CUDA
using DataFrames
using MCMCDepth
using PoseErrors
using Random
using SciGL

using Logging
using ProgressLogging
using TerminalLoggers
global_logger(TerminalLogger(right_justify=120))

CUDA.allowscalar(false)

# General experiment
experiment_name = "baseline"
result_dir = datadir("exp_raw", experiment_name)
parameters = Parameters()
sampler = [:mh_sampler, :mtm_sampler]
# only applicable to MTM
n_particles = 10
pose_time = 0.5
dataset = ["itodd", "lm", "tless"]
testset = "train_pbr"
scene_id = [0:4...]
configs = dict_list(@dict sampler n_particles pose_time dataset testset scene_id)

"""
    rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row, sampler_symbol)
Assembles the posterior model and the sampler from the loaded images, mesh, and DataFrame row.
"""
function rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row, sampler_symbol)
    # Context
    cpu_rng = Random.default_rng(parameters)
    dev_rng = device_rng(parameters)

    # Setup experiment
    camera = crop_camera(df_row)
    prior_o = fill(parameters.float_type(parameters.o_mask_not), parameters.width, parameters.height) |> device_array_type(parameters)
    prior_o[mask_img] .= parameters.o_mask_is
    # Prior t from mask is imprecise no need to bias
    prior_t = point_from_segmentation(df_row.bbox, depth_img, mask_img, df_row.cv_camera)
    experiment = Experiment(gl_context, Scene(camera, [mesh]), prior_o, prior_t, depth_img)

    # Model
    prior = point_prior(parameters, experiment, cpu_rng)
    posterior = simple_posterior(parameters, experiment, prior, dev_rng)
    # Sampler
    sampler = eval(sampler_symbol)(cpu_rng, parameters, posterior)
    # Result
    cpu_rng, posterior, sampler
end

"""
    timed_inference
Report the wall time from the point right after the raw data (the image, 3D object models etc.) is loaded.
"""
function timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row, sampler_symbol)
    time = @elapsed begin
        rng, posterior, sampler = rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row, sampler_symbol)
        # Sampling
        chain = sample(rng, posterior, sampler, parameters.n_steps; discard_initial=parameters.n_burn_in, thinning=parameters.n_thinning, progress=false)
        # Extract best pose and score
        score, idx = findmax(loglikelihood.(chain))
        t = variables(chain[idx]).t
        r = variables(chain[idx]).r
    end
    t, r, score, chain, time
end

"""
scene_inference(gl_context, config)
    Save results per scene via DrWatson's produce_or_load for the `config`
"""
function scene_inference(gl_context, config)
    # Extract config and load dataset
    @unpack dataset, testset, scene_id, n_particles, pose_time, sampler = config
    sampler_symbol = sampler
    scene_df = bop_test_or_train(dataset, testset, scene_id)
    parameters = Parameters()
    @reset parameters.n_particles = n_particles

    # Store result in DataFrame. Numerical precision doesn't matter here → Float32
    result_df = select(scene_df, :scene_id, :img_id, :obj_id)
    result_df.score = Vector{Float32}(undef, nrow(result_df))
    result_df.R = Vector{Quaternion{Float32}}(undef, nrow(result_df))
    result_df.t = Vector{Vector{Float32}}(undef, nrow(result_df))
    result_df.time = Vector{Float32}(undef, nrow(result_df))
    result_df.chain = Vector{Vector{Sample}}(undef, nrow(result_df))

    # Benchmark model sampler configuration to adapt number of steps - also avoids timing pre-compilation
    df_row = first(scene_df)
    depth_img, mask_img, mesh = load_img_mesh(df_row, parameters, gl_context)
    rng, posterior, sampler = rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row, sampler_symbol)
    step_time = mean_step_time(rng, posterior, sampler)
    @reset parameters.n_steps = floor(Int, pose_time / step_time)

    # Run inference per detection
    @progress "sampler: $(sampler_symbol), scene_id: $scene_id" for (idx, df_row) in enumerate(eachrow(scene_df))
        # Image crops differ per object
        depth_img, mask_img, mesh = load_img_mesh(df_row, parameters, gl_context)
        # Run and collect results
        t, R, score, chain, time = timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row, sampler_symbol)
        # Avoid out of GPU errors and save storage by not storing μ and o
        chain = collect_variables(chain, (:t, :r))
        result_df[idx, :].score = score
        result_df[idx, :].R = R
        result_df[idx, :].t = t
        result_df[idx, :].time = time
        result_df[idx, :].chain = chain
    end
    # Return result
    @strdict parameters result_df
end

# Avoid recreating the context in scene_inference by conditioning on it / closure
gl_context = render_context(parameters)
gl_scene_inference = scene_inference | gl_context
@progress "MCMC baseline" for config in configs
    @produce_or_load(gl_scene_inference, config, result_dir; filename=my_savename)
end
destroy_context(gl_context)

# Calculate errors
evaluate_errors(experiment_name)
