# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
Run different SMC algorithms on the synthetic BOP datasets:
* MCMC
* Forward Proposals
* Bootstrap

Model setup for segmentation:
* Segmentation prior for position `t` and pixel association `o`
* Simple likelihood function with mixture model for the pixels, a simple regularization, and without modeling the association,
"""

using DrWatson
@quickactivate("MCMCDepth")

@info "Loading packages"
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

"""
    parameters_and_sampler(sampler)
Parameters are hand-tuned for good results at ~0.5s per inference.

Returns (parameters, eval(sampler))
"""
function parameters_and_sampler(sampler)
    parameters = Parameters()
    # For simple_posterior
    @reset parameters.c_reg = 1 / 500
    if sampler == :smc_bootstrap
        @reset parameters.n_particles = 250
        @reset parameters.n_steps = 150
    elseif sampler == :smc_forward
        @reset parameters.n_particles = 250
        @reset parameters.n_steps = 150
    elseif sampler == :smc_mh
        @reset parameters.n_particles = 100
        @reset parameters.n_steps = 200
    end
    parameters, eval(sampler)
end

"""
    timed_inference
Report the wall time from the point right after the raw data (the image, 3D object models etc.) is loaded.
"""
function timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row, sampler)
    # Context
    cpu_rng = Random.default_rng(parameters)
    dev_rng = device_rng(parameters)

    time = @elapsed begin
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
        sampler = sampler(cpu_rng, parameters, posterior)
        states, final_state = smc_inference(cpu_rng, posterior, sampler, parameters)

        # Extract best pose and score
        sample = final_state.sample
        score, idx = findmax(loglikelihood(sample))
        t = variables(sample).t[:, idx]
        r = variables(sample).r[idx]
    end
    t, r, score, final_state, states, time
end

"""
scene_inference(gl_context, config)
    Save results per scene via DrWatson's produce_or_load for the `config`
"""
function scene_inference(gl_context, config)
    # Extract config and load dataset
    @unpack scene_id, dataset, testset, sampler = config
    scene_df = bop_test_or_train(dataset, testset, scene_id)
    parameters, sampler = parameters_and_sampler(sampler)

    # Store result in DataFrame. Numerical precision doesn't matter here → Float32
    result_df = select(scene_df, :scene_id, :img_id, :obj_id)
    result_df.score = Vector{Float32}(undef, nrow(result_df))
    result_df.R = Vector{Quaternion{Float32}}(undef, nrow(result_df))
    result_df.t = Vector{Vector{Float32}}(undef, nrow(result_df))
    result_df.time = Vector{Float32}(undef, nrow(result_df))
    result_df.final_state = Vector{SmcState}(undef, nrow(result_df))
    result_df.log_evidence = Vector{Vector{Float32}}(undef, nrow(result_df))

    # Avoid timing the pre-compilation
    df_row = first(scene_df)
    depth_img, mask_img, mesh = load_img_mesh(df_row, parameters, gl_context)
    timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row, sampler)

    # Run inference per detection
    @progress "sampler: $sampler, scene_id: $scene_id" for (idx, df_row) in enumerate(eachrow(scene_df))
        # Image crops differ per object
        depth_img, mask_img, mesh = load_img_mesh(df_row, parameters, gl_context)
        # Run and collect results
        t, R, score, final_state, states, time = timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row, sampler)
        # Avoid too large files by only saving t, r, and the logevidence not the sequence of states
        final_state = collect_variables(final_state, (:t, :r))
        # Avoid out of GPU errors
        @reset final_state.sample = to_cpu(final_state.sample)
        result_df[idx, :].score = score
        result_df[idx, :].R = R
        result_df[idx, :].t = t
        result_df[idx, :].time = time
        result_df[idx, :].final_state = final_state
        result_df[idx, :].log_evidence = logevidence.(states)
    end

    @strdict parameters result_df
end

experiment_name = "baseline"
gl_context = render_context(Parameters())
# Avoid recreating the context in scene_inference by conditioning on it / closure
gl_scene_inference = scene_inference | gl_context

# bop_datasets = [("lmo", "test"), ("tless", "test_primesense"), ("itodd", "val")]
bop_datasets = [("itodd", "train_pbr"), ("lm", "train_pbr"), ("tless", "train_pbr")]
@info "Run SMC on datasets $bop_datasets"
@progress "SMC baseline" for bop_dataset in bop_datasets
    # DrWatson configuration
    dataset, testset = bop_dataset
    bop_full_path = datadir("bop", bop_dataset...)
    scene_id = bop_scene_ids(bop_full_path)
    sampler = [:smc_mh, :smc_forward, :smc_bootstrap]
    config = @dict dataset testset scene_id sampler
    dicts = dict_list(config)

    # Run and save results
    result_path = datadir("exp_raw", experiment_name)
    @progress "dataset: $bop_dataset" for d in dicts
        @produce_or_load(gl_scene_inference, d, result_path; filename=my_savename)
    end
end
destroy_context(gl_context)

# Calculate errors
evaluate_errors(experiment_name)
