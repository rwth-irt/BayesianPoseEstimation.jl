# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
Run different SMC-MH on the synthetic BOP datasets for varying resolutions.
Keep constant:
* Number of steps
* Inference time
"""

# BUG for some reason, sometimes all particles have the same log likelihood for large resolutions. Check result plots and delete the faulty ones which have a very low recall.

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
scene_id = 0
resolution = [25:25:150...]
# Which one to keep constant
mode = [:time, :steps]
configs = dict_list(@dict dataset testset scene_id mode resolution)

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
        final_sample = final_state.sample
        score, idx = findmax(loglikelihood(final_sample))
        t = variables(final_sample).t[:, idx]
        r = variables(final_sample).r[idx]
    end
    t, r, score, final_state, states, time
end

"""
scene_inference(config)
    Save results per scene via DrWatson's produce_or_load for the `config`
"""
function scene_inference(config)
    # Extract config and load dataset
    @unpack dataset, testset, scene_id, resolution, mode = config
    scene_df = bop_test_or_train(dataset, testset, scene_id)
    parameters = Parameters()

    # Sampling parameters & OpenGL context
    @reset parameters.n_particles = 100
    @reset parameters.n_steps = 200
    @reset parameters.width = resolution
    @reset parameters.height = resolution
    @reset parameters.depth = parameters.n_particles
    gl_context = render_context(parameters)
    # Finally destroy gl_context
    try
        if mode == :time
            df_row = first(scene_df)
            depth_img, mask_img, mesh = load_img_mesh(df_row, parameters, gl_context)
            step_time = mean_step_time(gl_context, parameters, depth_img, mask_img, mesh, df_row)
            @reset parameters.n_steps = floor(Int, 0.5 / step_time)
        end

        # Store result in DataFrame. Numerical precision doesn't matter here → Float32
        result_df = select(scene_df, :scene_id, :img_id, :obj_id)
        result_df.score = Vector{Float32}(undef, nrow(result_df))
        result_df.R = Vector{Quaternion{Float32}}(undef, nrow(result_df))
        result_df.t = Vector{Vector{Float32}}(undef, nrow(result_df))
        result_df.time = Vector{Float32}(undef, nrow(result_df))
        result_df.final_state = Vector{SmcState}(undef, nrow(result_df))
        result_df.log_evidence = Vector{Vector{Float32}}(undef, nrow(result_df))

        # Run inference per detection
        @progress "dataset: $dataset, scene_id: $scene_id, resolution: $resolution, mode: $mode" for (idx, df_row) in enumerate(eachrow(scene_df))
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
    @produce_or_load(scene_inference, config, result_dir; filename=my_savename)
end

# Calculate errors
evaluate_errors(experiment_name)

# Combine results per resolution and mode
function parse_config(path)
    config = my_parse_savename(path)
    @unpack resolution, mode = config
    resolution, mode
end
pro_df = collect_results(datadir("exp_pro", experiment_name, "errors"))
transform!(pro_df, :path => ByRow(parse_config) => [:resolution, :mode])

# Threshold errors
transform!(pro_df, :adds => ByRow(x -> threshold_errors(x, ADDS_θ)) => :adds_thresh)
transform!(pro_df, :vsd => ByRow(x -> threshold_errors(x, BOP18_θ)) => :vsd_thresh)
transform!(pro_df, :vsdbop => ByRow(x -> threshold_errors(vcat(x...), BOP19_THRESHOLDS)) => :vsdbop_thresh)

# Recalls by resolution & mode
recall_groups = groupby(pro_df, [:resolution, :mode])
recalls = combine(recall_groups, :adds_thresh => (x -> recall(x...)) => :adds_recall, :vsd_thresh => (x -> recall(x...)) => :vsd_recall, :vsdbop_thresh => (x -> recall(x...)) => :vsdbop_recall)

# Raw results for n_steps and mean_time
raw_df = collect_results(datadir("exp_raw", experiment_name))
transform!(raw_df, :path => ByRow(parse_config) => [:resolution, :mode])
ts_groups = groupby(raw_df, [:resolution, :mode])
times_and_steps = combine(ts_groups, :result_df => (rdf -> mean(vcat(getproperty.(rdf, :time)...))) => :mean_time, :parameters => (p -> getproperty.(p, :n_steps) |> mean) => :mean_steps)
# Sanity check of inference times and steps
display(times_and_steps)

# Visualize
using Plots
gr()
diss_defaults()

# steps mode
steps_recalls = filter(:mode => x -> x == "steps", recalls)
steps_time = filter(:mode => x -> x == "steps", times_and_steps)
sort!(steps_recalls, :resolution)
sort!(steps_time, :resolution)

p1 = plot(steps_recalls.resolution, steps_recalls.adds_recall; label="ADDS", xlabel="resolution / px", ylabel="recall", ylims=[0, 1], linewidth=1.5, legend=:right)
plot!(steps_recalls.resolution, steps_recalls.vsd_recall; label="VSD", linewidth=1.5)
plot!(steps_recalls.resolution, steps_recalls.vsdbop_recall; label="VSDBOP", linewidth=1.5)
plot!(twinx(), steps_recalls.resolution, steps_time.mean_time; ylabel="pose inference time / s", label="inference time", linestyle=:dash, color=:black, legend=:bottomright)

display(p1)
savefig(p1, joinpath("plots", "$(experiment_name)_steps.pdf"))

# times mode
time_recalls = filter(:mode => x -> x == "time", recalls)
time_steps = filter(:mode => x -> x == "time", times_and_steps)
sort!(time_recalls, :resolution)
sort!(time_steps, :resolution)

p1 = plot(time_recalls.resolution, time_recalls.adds_recall; label="ADDS", xlabel="resolution / px", ylabel="recall", ylims=[0, 1], linewidth=1.5, legend=:left)
plot!(time_recalls.resolution, time_recalls.vsd_recall; label="VSD", linewidth=1.5)
plot!(time_recalls.resolution, time_recalls.vsdbop_recall; label="VSDBOP", linewidth=1.5)
plot!(twinx(), time_recalls.resolution, time_steps.mean_steps; ylabel="pose inference steps", ylimits=[0, Inf], label="steps", linestyle=:dash, color=:black, legend=:bottomleft)

display(p1)
savefig(p1, joinpath("plots", "$(experiment_name)_steps.pdf"))
