# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
Results from "Results for Simulated Instruments"
"""

using DrWatson
@quickactivate("MCMCDepth")

using Accessors
using CUDA
using DataFrames
using MCMCDepth
using PoseErrors
using Logging
using Random
using SciGL
using Statistics

using ProgressLogging
using TerminalLoggers
using Logging: global_logger
global_logger(TerminalLogger(right_justify=120))

CUDA.allowscalar(false)

# General experiment
experiment_name = "smc_steri"
result_dir = datadir("exp_raw", experiment_name)
dataset = "steri"
testset = "train_pbr"
scene_id = [1:9...]
sampler = :smc_mh
n_particles = 100 # [50, 100]
pose_time = [0.05, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
configs = dict_list(@dict dataset testset scene_id n_particles pose_time sampler)

parameters = Parameters()
@reset parameters.o_mask_is = 0.99
@reset parameters.o_mask_not = 1 - parameters.o_mask_is
@reset parameters.pixel_σ = 0.006587158392294226
@reset parameters.proposal_σ_r = fill(0.105350858684882, 3)
@reset parameters.width = 60
@reset parameters.height = 60
@reset parameters.depth = maximum(n_particles)

"""
    rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row, sampler_symbol)
Assembles the posterior model and the sampler from the loaded images, mesh, and DataFrame row.
"""
function rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row, sampler_symbol)
    # Context
    cpu_rng = Random.default_rng(parameters)
    dev_rng = device_rng(parameters)

    camera = crop_camera(df_row)
    prior_o = fill(parameters.float_type(parameters.o_mask_not), parameters.width, parameters.height) |> device_array_type(parameters)
    prior_o[mask_img] .= parameters.float_type(parameters.o_mask_is)
    # run without masks
    # prior_o = parameters.float_type(0.5)
    # Bias the point prior
    prior_t = df_row.gt_t + rand(KernelNormal(0, first(parameters.σ_t)), 3)
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
    timed = @elapsed begin
        rng, posterior, sampler = rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row, sampler_symbol)

        _, final_state = smc_inference(rng, posterior, sampler, parameters)
        # Extract best pose and score
        final_sample = final_state.sample
        score, idx = findmax(loglikelihood(final_sample))
        t = variables(final_sample).t[:, idx]
        r = variables(final_sample).r[idx]
    end
    t, r, score, timed
end

"""
scene_inference(gl_context, parameters, config)
    Save results per scene via DrWatson's produce_or_load for the `config`
"""
function scene_inference(gl_context, parameters, config)
    # Extract config and load dataset
    @unpack dataset, testset, scene_id, pose_time, n_particles, sampler = config
    sampler_symbol = sampler
    scene_df = bop_test_or_train(dataset, testset, scene_id)
    @reset parameters.n_particles = n_particles
    @reset parameters.time_budget = pose_time

    # Store result in DataFrame. Numerical precision doesn't matter here → Float32
    result_df = select(scene_df, :scene_id, :img_id, :obj_id)
    result_df.score = Vector{Float32}(undef, nrow(result_df))
    result_df.R = Vector{Quaternion{Float32}}(undef, nrow(result_df))
    result_df.t = Vector{Vector{Float32}}(undef, nrow(result_df))
    result_df.time = Vector{Float32}(undef, nrow(result_df))

    # Benchmark model sampler configuration to adapt number of steps - also avoids timing pre-compilation
    df_row = first(scene_df)
    depth_img, mask_img, mesh = load_img_mesh(df_row, parameters, gl_context)
    rng, posterior, sampler = rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row, sampler_symbol)
    step_time = mean_step_time(rng, posterior, sampler)
    @reset parameters.n_steps = floor(Int, pose_time / step_time)
    # For slow systems
    if parameters.n_steps < 2
        @reset parameters.n_steps = 2
    end

    # Run inference per detection
    @progress "sampler: $(sampler_symbol), n_particles: $(n_particles)" for (idx, df_row) in enumerate(eachrow(scene_df))
        # Image crops differ per object
        depth_img, mask_img, mesh = load_img_mesh(df_row, parameters, gl_context)
        # Run and collect results
        t, R, score, timed = timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row, sampler_symbol)
        result_df[idx, :].score = score
        result_df[idx, :].R = R
        result_df[idx, :].t = t
        result_df[idx, :].time = timed
    end
    @strdict parameters result_df
end

# OpenGL context - avoid recreating the context in scene_inference by using it in a closure
gl_context = render_context(parameters)
gl_scene_inference = scene_inference | (gl_context, parameters)
@progress "SMC Benchmark" for config in configs
    @produce_or_load(gl_scene_inference, config, result_dir; filename=c -> savename(c; connector=","))
end
destroy_context(gl_context)

# Calculate errors
evaluate_errors(experiment_name)

# Plot
import CairoMakie as MK
diss_defaults()

function parse_config(path)
    config = my_parse_savename(path)
    @unpack pose_time, n_particles, sampler = config
    pose_time, n_particles, sampler
end

# Calculate recalls
pro_df = collect_results(datadir("exp_pro", experiment_name, "errors"))
transform!(pro_df, :path => ByRow(parse_config) => [:pose_time, :n_particles, :sampler])
filter!(x -> x.n_particles > 1, pro_df)
# Threshold errors
transform!(pro_df, :adds => ByRow(x -> threshold_errors(x, ADDS_θ)) => :adds_thresh)
transform!(pro_df, :vsd => ByRow(x -> threshold_errors(x, BOP18_θ)) => :vsd_thresh)
transform!(pro_df, :vsdbop => ByRow(x -> threshold_errors(vcat(x...), BOP19_THRESHOLDS)) => :vsdbop_thresh)
groups = groupby(pro_df, [:sampler, :pose_time, :n_particles])
recalls = combine(groups, :adds_thresh => (x -> recall(x...)) => :adds_recall, :vsd_thresh => (x -> recall(x...)) => :vsd_recall, :vsdbop_thresh => (x -> recall(x...)) => :vsdbop_recall)

# Calculate mean pose inference times
raw_df = collect_results(result_dir)
transform!(raw_df, :path => ByRow(parse_config) => [:pose_time, :n_particles, :sampler])
times_groups = groupby(raw_df, [:sampler, :pose_time, :n_particles])
times = combine(times_groups, :result_df => (x -> mean(vcat(getproperty.(x, :time)...))) => :mean_time)

# Actually plot it
function plot_sampler(sampler_name, recalls, times)
    recalls_filtered = filter(x -> x.sampler == sampler_name, recalls)
    times_filtered = filter(x -> x.sampler == sampler_name, times)
    sort!(recalls_filtered, [:n_particles, :pose_time])
    sort!(times_filtered, [:n_particles, :pose_time])
    # Visualize per n_particles
    recall_groups = groupby(recalls_filtered, :n_particles)
    time_groups = groupby(times_filtered, :n_particles)

    fig = MK.Figure(resolution=(DISS_WIDTH, 0.5 * DISS_WIDTH); figure_padding=10)
    ax_vsd = MK.Axis(fig[2, 1]; xlabel="pose inference time / s", ylabel="recall / -", title="VSD", limits=(nothing, (0, 1)), yticks=0:0.25:1)
    for (rec, tim) in zip(recall_groups, time_groups)
        MK.lines!(ax_vsd, tim.mean_time, rec.vsd_recall; label="$(rec.n_particles |> first) particles")
    end
    MK.vlines!(ax_vsd, [0.5]; color=:black, linestyle=:dash)

    ax_adds = MK.Axis(fig[2, 2], xlabel="pose inference time / s", ylabel="recall / -", title="ADDS", limits=(nothing, (0, 1)), yticks=0:0.25:1)
    # Lines   
    for (rec, tim) in zip(recall_groups, time_groups)
        MK.lines!(ax_adds, tim.mean_time, rec.adds_recall; label="$(rec.n_particles |> first) particles")
    end
    MK.vlines!(ax_adds, [0.5]; color=:black, linestyle=:dash)

    ga = fig[1, :] = MK.GridLayout()
    ax_vsdbop = MK.Axis(ga[1, 1]; xlabel="pose inference time / s", ylabel="recall / -", title="VSDBOP", limits=(nothing, (0, 1)), yticks=0:0.25:1)
    for (rec, tim) in zip(recall_groups, time_groups)
        MK.lines!(ax_vsdbop, tim.mean_time, rec.vsdbop_recall; label="$(rec.n_particles |> first) particles")
    end
    MK.vlines!(ax_vsdbop, [0.5]; color=:black, linestyle=:dash)
    MK.Legend(ga[1, 2], ax_vsdbop)

    # display(fig)
    save(joinpath("plots", "$(experiment_name)_$(sampler_name).pdf"), fig)
end

plot_sampler("smc_mh", recalls, times)
