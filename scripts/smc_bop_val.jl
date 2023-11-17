# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
Run the algorithms on validation and test sets using ground truth masks in "Benchmark for 6D Object Pose Estimation:
Results on Real Data"
"""

using DrWatson
@quickactivate("MCMCDepth")

import CairoMakie as MK
using Accessors
using CUDA
using CSV
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

# Combine results by sampler & dataset
baseline_res = collect_results(datadir("exp_pro", "baseline_hyperopt", "errors"))
directory = datadir("exp_pro", experiment_name, "errors")
validation_res = collect_results(directory)

function parse_config(path)
    config = my_parse_savename(path)
    @unpack sampler, dataset, scene_id = config
    sampler, dataset, scene_id
end
DataFrames.transform!(baseline_res, :path => ByRow(parse_config) => [:sampler, :dataset, :scene_id])
DataFrames.transform!(validation_res, :path => ByRow(parse_config) => [:sampler, :dataset, :scene_id])
# only compare smc samplers
subset(baseline_res, :sampler => x -> x .== "smc_mh")

# Threshold the errors
DataFrames.transform!(baseline_res, :adds => ByRow(x -> threshold_errors(x, ADDS_θ)) => :adds_thresh)
DataFrames.transform!(baseline_res, :vsd => ByRow(x -> threshold_errors(x, BOP18_θ)) => :vsd_thresh)
DataFrames.transform!(baseline_res, :vsdbop => ByRow(x -> threshold_errors(vcat(x...), BOP19_THRESHOLDS)) => :vsdbop_thresh)
DataFrames.transform!(validation_res, :adds => ByRow(x -> threshold_errors(x, ADDS_θ)) => :adds_thresh)
DataFrames.transform!(validation_res, :vsd => ByRow(x -> threshold_errors(x, BOP18_θ)) => :vsd_thresh)
DataFrames.transform!(validation_res, :vsdbop => ByRow(x -> threshold_errors(vcat(x...), BOP19_THRESHOLDS)) => :vsdbop_thresh)

# Recall by sampler
baseline_groups = groupby(baseline_res, [:sampler])
baseline_recalls = combine(baseline_groups, :adds_thresh => (x -> recall(x...)) => :adds_recall, :vsd_thresh => (x -> recall(x...)) => :vsd_recall, :vsdbop_thresh => (x -> recall(x...)) => :vsdbop_recall)

# Recall by sampler and dataset
validation_groups = groupby(validation_res, [:sampler, :dataset])
validation_recalls = combine(validation_groups, :adds_thresh => (x -> recall(x...)) => :adds_recall, :vsd_thresh => (x -> recall(x...)) => :vsd_recall, :vsdbop_thresh => (x -> recall(x...)) => :vsdbop_recall)
CSV.write(datadir("exp_pro", experiment_name, "sampler_dataset_recall.csv"), validation_recalls)
display(validation_recalls)

# Plot recall over error threshold
diss_defaults()

fig_recall = MK.Figure()
ax_vsd_recall = MK.Axis(fig_recall[2, 1]; xlabel="error threshold / -", ylabel="recall / -", limits=(nothing, (0, 1)), title="VSD")
ax_adds_recall = MK.Axis(fig_recall[2, 2]; xlabel="error threshold / -", ylabel="recall / -", limits=(nothing, (0, 1)), title="ADDS")
gl_recall = fig_recall[1, :] = MK.GridLayout()
ax_vsdbop_recall = MK.Axis(gl_recall[1, 1]; xlabel="error threshold / -", ylabel="recall / -", limits=(nothing, (0, 1)), title="VSDBOP")

fig_density = MK.Figure(figure_padding=10)
ax_vsd_density = MK.Axis(fig_density[2, 1]; xlabel="normalized error / -", ylabel="density / -", title="VSD")
ax_adds_density = MK.Axis(fig_density[2, 2]; xlabel="normalized error / -", ylabel="density / -", title="ADDS")
gl_density = fig_density[1, :] = MK.GridLayout()
ax_vsdbop_density = MK.Axis(gl_density[1, 1]; xlabel="normalized error / -", ylabel="density / -", title="VSDBOP")
θ_range = 0:0.02:1

# Plot synthetic baseline
adds_thresh = map(θ -> threshold_errors.(baseline_res.adds, θ), θ_range)
adds_recalls = map(x -> recall(x...), adds_thresh)
MK.lines!(ax_adds_recall, θ_range, adds_recalls; label="synthetic")
MK.density!(ax_adds_density, vcat(baseline_res.adds...); label="synthetic", boundary=(0, 1))

vsd_thresh = map(θ -> threshold_errors.(baseline_res.vsd, θ), θ_range)
vsd_recalls = map(x -> recall(x...), vsd_thresh)
MK.lines!(ax_vsd_recall, θ_range, vsd_recalls; label="synthetic")
MK.density!(ax_vsd_density, vcat(baseline_res.vsd...); label="synthetic", boundary=(0, 1))

vsdbop_thresh = map(θ -> threshold_errors.(vcat(baseline_res.vsdbop...), θ), θ_range)
vsdbop_recalls = map(x -> recall(x...), vsdbop_thresh)
MK.lines!(ax_vsdbop_recall, θ_range, vsdbop_recalls; label="synthetic")
MK.density!(ax_vsdbop_density, reduce(vcat, reduce(vcat, baseline_res.vsdbop)); label="synthetic", boundary=(0, 1))

# Plot validation
adds_thresh = map(θ -> threshold_errors.(validation_res.adds, θ), θ_range)
adds_recalls = map(x -> recall(x...), adds_thresh)
MK.lines!(ax_adds_recall, θ_range, adds_recalls; label="validation")
MK.density!(ax_adds_density, vcat(validation_res.adds...); label="validation", boundary=(0, 1))

vsd_thresh = map(θ -> threshold_errors.(validation_res.vsd, θ), θ_range)
vsd_recalls = map(x -> recall(x...), vsd_thresh)
MK.lines!(ax_vsd_recall, θ_range, vsd_recalls; label="validation")
MK.density!(ax_vsd_density, vcat(validation_res.vsd...); label="validation", boundary=(0, 1))

vsdbop_thresh = map(θ -> threshold_errors.(vcat(validation_res.vsdbop...), θ), θ_range)
vsdbop_recalls = map(x -> recall(x...), vsdbop_thresh)
MK.lines!(ax_vsdbop_recall, θ_range, vsdbop_recalls; label="validation")
MK.density!(ax_vsdbop_density, reduce(vcat, reduce(vcat, validation_res.vsdbop)); label="validation", boundary=(0, 1))

MK.vlines!(ax_vsdbop_recall, BOP19_THRESHOLDS)
MK.vspan!(ax_vsdbop_recall, 0, last(BOP19_THRESHOLDS))
MK.vlines!(ax_vsd_recall, BOP18_θ)
MK.vspan!(ax_vsd_recall, 0, BOP18_θ)
MK.vlines!(ax_adds_recall, ADDS_θ)
MK.vspan!(ax_adds_recall, 0, ADDS_θ)
MK.Legend(gl_recall[1, 2], ax_vsdbop_recall)
display(fig_recall)
save(joinpath("plots", "$(experiment_name)_recall.pdf"), fig_recall)

MK.vlines!(ax_vsdbop_density, BOP19_THRESHOLDS)
MK.vspan!(ax_vsdbop_density, 0, last(BOP19_THRESHOLDS))
MK.vlines!(ax_vsd_density, BOP18_θ)
MK.vspan!(ax_vsd_density, 0, BOP18_θ)
MK.vlines!(ax_adds_density, ADDS_θ)
MK.vspan!(ax_adds_density, 0, ADDS_θ)
MK.Legend(gl_density[1, 2], ax_vsdbop_density)
display(fig_density)
save(joinpath("plots", "$(experiment_name)_density.pdf"), fig_density)
