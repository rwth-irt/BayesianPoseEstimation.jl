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
@quickactivate

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

CUDA.allowscalar(false)

experiment_name = "smc_mh_resolution"
result_dir = datadir("exp_raw", experiment_name)
dataset = ["lm", "tless", "itodd"]
testset = "train_pbr"
scene_id = 0
resolution = [2, 5:5:25..., 30, 40, 50:25:100...]
# Which one to keep constant
mode = :time # [:time, :steps]
configs = dict_list(@dict dataset testset scene_id mode resolution)

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
    prior_o = parameters.float_type(0.5)
    # Bias the point prior
    prior_t = df_row.gt_t + rand(KernelNormal(0, first(parameters.σ_t)), 3)
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
            # Benchmark model sampler configuration
            df_row = first(scene_df)
            depth_img, mask_img, mesh = load_img_mesh(df_row, parameters, gl_context)
            rng, posterior, sampler = rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row)
            step_time = mean_step_time(rng, posterior, sampler)
            @reset parameters.n_steps = floor(Int, parameters.time_budget / step_time)
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
import CairoMakie as MK
diss_defaults()

# Const n_steps
if "steps" in recalls.mode
    steps_recalls = filter(:mode => x -> x == "steps", recalls)
    steps_time = filter(:mode => x -> x == "steps", times_and_steps)
    sort!(steps_recalls, :resolution)
    sort!(steps_time, :resolution)

    f1 = MK.Figure(size=(DISS_WIDTH, 2 / 5 * DISS_WIDTH))
    ax1 = MK.Axis(f1[1, 1], xlabel="resolution / px", ylabel="recall / -", xticks=steps_recalls.resolution, yticks=0:0.2:1, limits=(nothing, nothing, 0, 1))
    MK.lines!(ax1, steps_recalls.resolution, steps_recalls.adds_recall; label="ADDS")
    MK.lines!(ax1, steps_recalls.resolution, steps_recalls.vsd_recall; label="VSD")
    MK.lines!(ax1, steps_recalls.resolution, steps_recalls.vsdbop_recall; label="VSDBOP")
    ax2 = MK.Axis(f1[1, 1]; ylabel="pose inference time / s", yaxisposition=:right, xticksvisible=false, xticklabelsvisible=false, xgridvisible=false, ygridvisible=false, limits=(nothing, (0, 1)))
    MK.lines!(ax2, steps_recalls.resolution, steps_time.mean_time; label="avg. inference time", linestyle=:dash, color=:black)
    MK.axislegend(ax1; position=:rb)
    MK.axislegend(ax2; position=:cb)

    # display(f1)
    save(joinpath("plots", "$(experiment_name)_const_steps.pdf"), f1)
end

# Const time
if "time" in recalls.mode
    time_recalls = filter(:mode => x -> x == "time", recalls)
    time_steps = filter(:mode => x -> x == "time", times_and_steps)
    sort!(time_recalls, :resolution)
    sort!(time_steps, :resolution)

    f2 = MK.Figure(size=(DISS_WIDTH, 2 / 5 * DISS_WIDTH))
    ax1 = MK.Axis(f2[1, 1], xlabel="resolution / px", ylabel="recall / s", xticks=time_recalls.resolution, yticks=0:0.2:1, limits=(nothing, (0, 1)))
    MK.lines!(ax1, time_recalls.resolution, time_recalls.adds_recall; label="ADDS")
    MK.lines!(ax1, time_recalls.resolution, time_recalls.vsd_recall; label="VSD")
    MK.lines!(ax1, time_recalls.resolution, time_recalls.vsdbop_recall; label="VSDBOP")
    ax2 = MK.Axis(f2[1, 1]; ylabel="steps per inference / -", yaxisposition=:right, xticksvisible=false, xticklabelsvisible=false, xgridvisible=false, ygridvisible=false, limits=(nothing, (0, 1)))
    MK.lines!(ax2, time_recalls.resolution, time_steps.mean_steps; label="avg. inference steps", linestyle=:dash, color=:black)
    MK.limits!(ax2, 0, nothing, 0, nothing)
    MK.axislegend(ax1; position=:rb)
    MK.axislegend(ax2; position=:cb)

    # display(f2)
    save(joinpath("plots", "$(experiment_name)_const_time.pdf"), f2)
end
