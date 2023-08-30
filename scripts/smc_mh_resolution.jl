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
using BenchmarkTools
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

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1
CUDA.allowscalar(false)

experiment_name = "smc_mh_resolution"
result_dir = datadir("exp_raw", experiment_name)
dataset = ["lm", "tless", "itodd"]
testset = "train_pbr"
scene_id = [0:4...]
resolution = [10, 20, 30]
configs = dict_list(@dict dataset testset scene_id resolution)


"""
    mean_step_time(cpu_rng, posterior, sampler)
Returns the mean time per step of the benchmark in seconds.
This allows an approximation of the number of steps given a constant compute budget.
"""
function mean_step_time(gl_context, parameters, depth_img, mask_img, mesh, df_row)
    # Context
    cpu_rng = Random.default_rng(parameters)
    dev_rng = device_rng(parameters)

    # Model
    camera = crop_camera(df_row)
    prior_o = fill(parameters.float_type(parameters.o_mask_not), parameters.width, parameters.height) |> device_array_type(parameters)
    prior_o[mask_img] .= parameters.o_mask_is
    prior_t = point_from_segmentation(df_row.bbox, depth_img, mask_img, df_row.cv_camera)
    experiment = Experiment(gl_context, Scene(camera, [mesh]), prior_o, prior_t, depth_img)
    prior = point_prior(parameters, experiment, cpu_rng)
    posterior = simple_posterior(parameters, experiment, prior, dev_rng)

    # Benchmark sampler
    sampler = smc_mh(cpu_rng, parameters, posterior)
    _, state = AbstractMCMC.step(cpu_rng, posterior, sampler)
    # Interpolate local variables into the benchmark expression
    t = @benchmark AbstractMCMC.step($cpu_rng, $posterior, $sampler, $state)
    # Convert from ns to seconds
    mean(t.times) * 1e-9
end

"""
    timed_inference
Report the wall time from the point right after the raw data (the image, 3D object models etc.) is loaded.
"""
function timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row)
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
        sampler = smc_mh(cpu_rng, parameters, posterior)
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
scene_inference(config)
    Save results per scene via DrWatson's produce_or_load for the `config`
"""
function scene_inference(config)
    # Extract config and load dataset
    @unpack dataset, testset, scene_id, resolution = config
    scene_df = bop_test_or_train(dataset, testset, scene_id)
    parameters = Parameters()
    # For simple_posterior
    @reset parameters.c_reg = 1 / 500

    # Sampling parameters & OpenGL context
    @reset parameters.n_particles = 100
    @reset parameters.n_steps = 200
    @reset parameters.width = resolution
    @reset parameters.height = resolution
    @reset parameters.depth = parameters.n_particles
    gl_context = render_context(parameters)
    # Finally destroy gl_context
    try
        # TODO Approximately same inference time for all configurations - does it make sense?
        df_row = first(scene_df)
        depth_img, mask_img, mesh = load_img_mesh(df_row, parameters, gl_context)
        step_time = mean_step_time(gl_context, parameters, depth_img, mask_img, mesh, df_row)
        @reset parameters.n_steps = floor(Int, 0.5 / step_time)

        # Store result in DataFrame. Numerical precision doesn't matter here → Float32
        result_df = select(scene_df, :scene_id, :img_id, :obj_id)
        result_df.score = Vector{Float32}(undef, nrow(result_df))
        result_df.R = Vector{Quaternion{Float32}}(undef, nrow(result_df))
        result_df.t = Vector{Vector{Float32}}(undef, nrow(result_df))
        result_df.time = Vector{Float32}(undef, nrow(result_df))
        result_df.final_state = Vector{SmcState}(undef, nrow(result_df))
        result_df.log_evidence = Vector{Vector{Float32}}(undef, nrow(result_df))

        # Run inference per detection
        @progress "dataset: $dataset, scene_id: $scene_id, resolution: $resolution" for (idx, df_row) in enumerate(eachrow(scene_df))
            # Image crops differ per object
            depth_img, mask_img, mesh = load_img_mesh(df_row, parameters, gl_context)
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
        return @strdict parameters result_df
    finally
        # If not destroyed, weird stuff happens
        destroy_context(gl_context)
    end
end

@progress "SMC-MH resolution" for config in configs
    @produce_or_load(scene_inference, config, result_dir; filename=c -> savename(c; connector=","))
end
