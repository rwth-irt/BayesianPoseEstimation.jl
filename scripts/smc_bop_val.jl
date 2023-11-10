# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
Run the algorithms on validation and test sets using ground truth masks
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
experiment_name = "smc_bop_val_hyperopt"
result_dir = datadir("exp_raw", experiment_name)
parameters = Parameters()
@reset parameters.n_particles = 100
@reset parameters.depth = parameters.n_particles
@reset parameters.o_mask_is = 0.9
@reset parameters.o_mask_not = 1 - parameters.o_mask_is
@reset parameters.pixel_σ = 0.005
@reset parameters.proposal_σ_r = fill(π, 3)
sampler = :smc_mh
# no default detections in val
dataset = "itodd"
testset = "val"
scene_id = 1
itodd_config = dict_list(@dict sampler dataset testset scene_id)

# no default detections
dataset = "lm"
testset = "test"
scene_id = [1:15...]
lm_config = dict_list(@dict sampler dataset testset scene_id)

dataset = "lmo"
testset = "test"
scene_id = 2
lmo_config = dict_list(@dict sampler dataset testset scene_id)

dataset = "tless"
testset = "test_primesense"
scene_id = [1:20...]
tless_config = dict_list(@dict sampler dataset testset scene_id)

configs = [itodd_config..., lmo_config..., tless_config...]

"""
    rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row)
Assembles the posterior model and the sampler from the loaded images, mesh, and DataFrame row.
"""
function rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row)
    # Context
    cpu_rng = Random.default_rng(parameters)
    dev_rng = device_rng(parameters)

    # Setup experiment
    camera = crop_camera(df_row)
    prior_o = fill(parameters.float_type(parameters.o_mask_not), parameters.width, parameters.height) |> device_array_type(parameters)
    prior_o[mask_img] .= parameters.float_type(parameters.o_mask_is)
    # Bias the point prior
    prior_t = point_from_segmentation(df_row.bbox, depth_img, mask_img, df_row.cv_camera)
    experiment = Experiment(gl_context, Scene(camera, [mesh]), prior_o, prior_t, depth_img)

    # Model
    prior = point_prior(parameters, experiment, cpu_rng)
    posterior = simple_posterior(parameters, experiment, prior, dev_rng)
    # Sampler
    sampler = smc_mh(cpu_rng, parameters, posterior)
    # Result
    cpu_rng, posterior, sampler
end

"""
    timed_inference
Report the wall time from the point right after the raw data (the image, 3D object models etc.) is loaded.
"""
function timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row)
    time = @elapsed begin
        # Assemble sampler and run inference
        rng, posterior, sampler = rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row)
        states, final_state = smc_inference(rng, posterior, sampler, parameters)

        # Extract best pose and score
        final_sample = final_state.sample
        score, idx = findmax(loglikelihood(final_sample))
        t = variables(final_sample).t[:, idx]
        r = variables(final_sample).r[idx]
    end
    t, r, score, final_state, states, time
end

"""
scene_inference(gl_context, config)
    Save results per scene via DrWatson's produce_or_load for the `config`
"""
function scene_inference(gl_context, parameters, config)
    # Extract config and load dataset
    @unpack scene_id, dataset, testset = config
    # TODO in test script
    scene_df = train_targets(datadir("bop", dataset, testset), scene_id)

    # Store result in DataFrame. Numerical precision doesn't matter here → Float32
    result_df = select(scene_df, :scene_id, :img_id, :obj_id)
    result_df.score = Vector{Float32}(undef, nrow(result_df))
    result_df.R = Vector{Quaternion{Float32}}(undef, nrow(result_df))
    result_df.t = Vector{Vector{Float32}}(undef, nrow(result_df))
    result_df.time = Vector{Float32}(undef, nrow(result_df))
    result_df.final_state = Vector{SmcState}(undef, nrow(result_df))
    result_df.log_evidence = Vector{Vector{Float32}}(undef, nrow(result_df))

    # Load data and setup sampler
    df_row = first(scene_df)
    depth_img = load_depth_image(df_row, parameters.img_size...) |> device_array_type(parameters)
    # TODO in test script
    # mask_img = load_segmentation(df_row, parameters.img_size...) |> device_array_type(parameters)
    mask_img = load_mask_image(df_row, parameters.img_size...) |> device_array_type(parameters)
    mesh = upload_mesh(gl_context, load_mesh(df_row))
    rng, posterior, sampler = rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row)
    # Benchmark model sampler configuration to adapt number of steps - also avoids timing pre-compilation
    step_time = mean_step_time(rng, posterior, sampler)
    @reset parameters.n_steps = floor(Int, parameters.time_budget / step_time)
    # For slow systems
    if parameters.n_steps < 2
        @reset parameters.n_steps = 2
    end

    # Run inference per detection
    @progress "scene_id: $scene_id" for (idx, df_row) in enumerate(eachrow(scene_df))
        # Image crops differ per object
        depth_img = load_depth_image(df_row, parameters.img_size...) |> device_array_type(parameters)
        # TODO in test script
        # mask_img = load_segmentation(df_row, parameters.img_size...) |> device_array_type(parameters)
        mask_img = load_mask_image(df_row, parameters.img_size...) |> device_array_type(parameters)
        mesh = upload_mesh(gl_context, load_mesh(df_row))
        # Run and collect results
        t, R, score, final_state, states, time = timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row)
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

# Avoid recreating the context in scene_inference by conditioning on it / closure
gl_context = render_context(parameters)
gl_scene_inference = scene_inference | (gl_context, parameters)
@progress "SMC BOP validation" for config in configs
    @produce_or_load(gl_scene_inference, config, result_dir; filename=my_savename)
end
destroy_context(gl_context)

# Calculate errors and recalls
evaluate_errors(experiment_name)
evaluate_recalls(experiment_name)
