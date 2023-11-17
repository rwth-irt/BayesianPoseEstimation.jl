# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
Test the influence of masks and point priors in "Choice of Priors for Position and Classfication"
"""

using DrWatson
@quickactivate("MCMCDepth")

using Accessors
using CSV
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

import CairoMakie as MK
diss_defaults()
CUDA.allowscalar(false)

# General experiment
experiment_name = "smc_priors"
result_dir = datadir("exp_raw", experiment_name)
dataset = ["lm", "tless", "itodd"]
prior = [:point, :mask, :point_mask]
testset = "train_pbr"
scene_id = [0:4...]
configs = dict_list(@dict dataset testset scene_id prior)

function experiment_for_symbol(prior_symbol, gl_context, parameters, df_row)
    depth_img, mask_img, mesh = load_img_mesh(df_row, parameters, gl_context)
    camera = crop_camera(df_row)
    if prior_symbol == :point
        # Flat prior for mask
        prior_o = parameters.o_mask_is
        # Position prior from RFID tag
        # NOTE no seeding here, always same noise is not wanted
        prior_t = df_row.gt_t + rand(KernelNormal(0, 0.005f0), 3)
    elseif prior_symbol == :mask
        # Use mask
        prior_o = fill(parameters.float_type(parameters.o_mask_not), parameters.width, parameters.height) |> device_array_type(parameters)
        prior_o[mask_img] .= parameters.o_mask_is
        # Position prior from mask
        prior_t = point_from_segmentation(df_row.bbox, depth_img, mask_img, df_row.cv_camera)
    elseif prior_symbol == :point_mask
        # Use mask
        prior_o = fill(parameters.float_type(parameters.o_mask_not), parameters.width, parameters.height) |> device_array_type(parameters)
        prior_o[mask_img] .= parameters.o_mask_is
        # Position prior from RFID tag
        # NOTE no seeding here, always same noise is not wanted
        prior_t = df_row.gt_t + rand(KernelNormal(0, 0.005f0), 3)
    end
    Experiment(gl_context, Scene(camera, [mesh]), prior_o, prior_t, depth_img)
end

"""
    rng_posterior_sampler(gl_context, parameters, df_row, prior_symbol)
Assembles the posterior model and sampler.
"""
function rng_posterior_sampler(gl_context, parameters, df_row, prior_symbol)
    # Context
    cpu_rng = Random.default_rng(parameters)
    dev_rng = device_rng(parameters)
    experiment = experiment_for_symbol(prior_symbol, gl_context, parameters, df_row)
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
function timed_inference(gl_context, parameters, df_row, prior_symbol)
    timed = @elapsed begin
        rng, posterior, sampler = rng_posterior_sampler(gl_context, parameters, df_row, prior_symbol)
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
scene_inference(gl_context, config)
    Save results per scene via DrWatson's produce_or_load for the `config`
"""
function scene_inference(gl_context, config)
    # Extract config and load dataset
    @unpack dataset, testset, scene_id, prior = config
    scene_df = bop_test_or_train(dataset, testset, scene_id)
    parameters = Parameters()

    # Store result in DataFrame. Numerical precision doesn't matter here → Float32
    result_df = select(scene_df, :scene_id, :img_id, :obj_id)
    result_df.score = Vector{Float32}(undef, nrow(result_df))
    result_df.R = Vector{Quaternion{Float32}}(undef, nrow(result_df))
    result_df.t = Vector{Vector{Float32}}(undef, nrow(result_df))
    result_df.time = Vector{Float32}(undef, nrow(result_df))

    # Benchmark model sampler configuration to adapt number of steps - also avoids timing pre-compilation
    df_row = first(scene_df)
    rng, posterior, sampler = rng_posterior_sampler(gl_context, parameters, df_row, prior)
    step_time = mean_step_time(rng, posterior, sampler)
    # Compute budget of 5sec
    @reset parameters.n_steps = floor(Int, parameters.time_budget / step_time)

    # Run inference per detection
    @progress "dataset: $(dataset), testset: $(testset), scene_id : $(scene_id)" for (idx, df_row) in enumerate(eachrow(scene_df))
        # Image crops differ per object
        depth_img, mask_img, mesh = load_img_mesh(df_row, parameters, gl_context)
        # Run and collect results
        t, R, score, timed = timed_inference(gl_context, parameters, df_row, prior)
        result_df[idx, :].score = score
        result_df[idx, :].R = R
        result_df[idx, :].t = t
        result_df[idx, :].time = timed
    end
    @strdict parameters result_df
end

# OpenGL context
parameters = Parameters()
# Avoid recreating the context in scene_inference by conditioning on it / closure
gl_context = render_context(parameters)
gl_scene_inference = scene_inference | gl_context
@progress "SMC priors" for config in configs
    @produce_or_load(gl_scene_inference, config, result_dir; filename=c -> savename(c; connector=","))
end
destroy_context(gl_context)

# Evaluate
evaluate_errors(experiment_name)
function parse_config(path)
    config = my_parse_savename(path)
    @unpack prior, dataset = config
    prior, dataset
end

# TODO put it in a function together with evaluate_errors
# Calculate recalls
pro_df = collect_results(datadir("exp_pro", experiment_name, "errors"))
transform!(pro_df, :path => ByRow(parse_config) => [:prior, :dataset])
# Threshold errors
transform!(pro_df, :adds => ByRow(x -> threshold_errors(x, ADDS_θ)) => :adds_thresh)
transform!(pro_df, :vsd => ByRow(x -> threshold_errors(x, BOP18_θ)) => :vsd_thresh)
transform!(pro_df, :vsdbop => ByRow(x -> threshold_errors(vcat(x...), BOP19_THRESHOLDS)) => :vsdbop_thresh)

# Recall by prior
groups = groupby(pro_df, [:prior])
recalls = combine(groups, :adds_thresh => (x -> recall(x...)) => :adds_recall, :vsd_thresh => (x -> recall(x...)) => :vsd_recall, :vsdbop_thresh => (x -> recall(x...)) => :vsdbop_recall)
CSV.write(datadir("exp_pro", experiment_name, "prior_recall.csv"), recalls)
display(recalls)

# Recall by prior and dataset
groups = groupby(pro_df, [:prior, :dataset])
recalls = combine(groups, :adds_thresh => (x -> recall(x...)) => :adds_recall, :vsd_thresh => (x -> recall(x...)) => :vsd_recall, :vsdbop_thresh => (x -> recall(x...)) => :vsdbop_recall)
CSV.write(datadir("exp_pro", experiment_name, "prior_dataset_recall.csv"), recalls)
display(recalls)

function fig_xtick(dataset)
    if dataset == "itodd"
        "ITODD"
    elseif dataset == "lm"
        "LM"
    elseif dataset == "tless"
        "T-LESS"
    end
end

function fig_label(group)
    prior = first(group.prior)
    if prior == "mask"
        "mask"
    elseif prior == "point"
        "point"
    elseif prior == "point_mask"
        "both"
    end
end

fig = MK.Figure(resolution=(DISS_WIDTH, 0.3 * DISS_WIDTH))
# Dataset names on x-axis
xnames = unique(pro_df.dataset) .|> fig_xtick
xticks = (eachindex(xnames), xnames)

ax = MK.Axis(fig[1, 1]; title="ADDS", xticks=xticks, ylabel="recall / -", limits=(nothing, (0.5, 1)))
groups = groupby(recalls, [:prior])
for group in groups
    group = sort!(group, :dataset)
    MK.scatterlines!(ax, group.adds_recall; label=fig_label(group))
end

ax = MK.Axis(fig[1, 2]; title="VSDBOP", xticks=xticks, limits=(nothing, (0.5, 1)))
groups = groupby(recalls, [:prior])
for group in groups
    group = sort!(group, :dataset)
    MK.scatterlines!(ax, group.vsdbop_recall; label=fig_label(group))
end

ax = MK.Axis(fig[1, 3]; title="VSD", xticks=xticks, limits=(nothing, (0.5, 1)))
groups = groupby(recalls, [:prior])
for group in groups
    group = sort!(group, :dataset)
    MK.scatterlines!(ax, group.vsd_recall; label=fig_label(group))
end

MK.Legend(fig[1, 4], ax)
save(joinpath("plots", "$(experiment_name).pdf"), fig)
# display(fig)

