# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
Run different Metropolis Hastings MCMC on  the synthetic BOP datasets.
Only the first scene of each dataset is evaluated because of the computation time.

WARN: Results vary based on sampler configuration
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
using Statistics

using Logging
using ProgressLogging
using TerminalLoggers
global_logger(TerminalLogger(right_justify=120))

CUDA.allowscalar(false)

# General experiment
experiment_name = "mcmc_benchmark"
result_dir = datadir("exp_raw", experiment_name)

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
    prior_o = parameters.float_type(0.5)
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
    timed = @elapsed begin
        rng, posterior, sampler = rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row, sampler_symbol)
        # Only MTM supports multiple particles. Metropolis Hastings in recall_n_steps.jl
        chain = sample(rng, posterior, sampler, parameters.n_steps; discard_initial=parameters.n_burn_in, thinning=parameters.n_thinning, progress=false)
        # Extract best pose and score
        score, idx = findmax(loglikelihood.(chain))
        t = variables(chain[idx]).t
        r = variables(chain[idx]).r
    end
    t, r, score, timed
end

"""
scene_inference(gl_context, config)
    Save results per scene via DrWatson's produce_or_load for the `config`
"""
function scene_inference(gl_context, config)
    # Extract config and load dataset
    @unpack dataset, testset, scene_id, pose_time, n_particles, sampler = config
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

# MH
sampler = :mh_sampler
n_particles = 1
dataset = ["lm", "tless", "itodd"]
testset = "train_pbr"
scene_id = 0
pose_time = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
configs = dict_list(@dict dataset testset scene_id n_particles pose_time sampler)
# OpenGL context
parameters = Parameters()
@reset parameters.depth = 1
gl_context = render_context(parameters)
gl_scene_inference = scene_inference | gl_context
# Run experiments
@progress "MH Benchmark" for config in configs
    @produce_or_load(gl_scene_inference, config, result_dir; filename=my_savename)
end
destroy_context(gl_context)

# MTM
sampler = :mtm_sampler
n_particles = [5, 10, 20, 40]
configs = dict_list(@dict dataset testset scene_id n_particles pose_time sampler)
# OpenGL context
parameters = Parameters()
@reset parameters.depth = maximum(n_particles)
gl_context = render_context(parameters)
gl_scene_inference = scene_inference | gl_context
# Run experiments
@progress "MTM Benchmark" for config in configs
    @produce_or_load(gl_scene_inference, config, result_dir; filename=my_savename)
end
destroy_context(gl_context)


# Calculate errors
evaluate_errors(experiment_name)

# Load results
function parse_config(path)
    config = my_parse_savename(path)
    @unpack pose_time, n_particles, sampler = config
    pose_time, n_particles, sampler
end

# Calc recalls and mean inference times
pro_df = collect_results(datadir("exp_pro", experiment_name, "errors"))
transform!(pro_df, :path => ByRow(parse_config) => [:pose_time, :n_particles, :sampler])
# Threshold errors
transform!(pro_df, :adds => ByRow(x -> threshold_errors(x, ADDS_θ)) => :adds_thresh)
transform!(pro_df, :vsd => ByRow(x -> threshold_errors(x, BOP18_θ)) => :vsd_thresh)
transform!(pro_df, :vsdbop => ByRow(x -> threshold_errors(vcat(x...), BOP19_THRESHOLDS)) => :vsdbop_thresh)
# Recall & time by inference time
recall_groups = groupby(pro_df, [:sampler, :pose_time, :n_particles])
recalls = combine(recall_groups, :adds_thresh => (x -> recall(x...)) => :adds_recall, :vsd_thresh => (x -> recall(x...)) => :vsd_recall, :vsdbop_thresh => (x -> recall(x...)) => :vsdbop_recall)
# Mean inference time
raw_df = collect_results(datadir("exp_raw", experiment_name))
transform!(raw_df, :path => ByRow(parse_config) => [:pose_time, :n_particles, :sampler])
times_groups = groupby(raw_df, [:sampler, :pose_time, :n_particles])
times = combine(times_groups, :result_df => (x -> mean(vcat(getproperty.(x, :time)...))) => :mean_time)

# Visualization
import CairoMakie as MK
diss_defaults()

# Visualize per n_particles
sort!(recalls, [:n_particles, :pose_time])
sort!(times, [:n_particles, :pose_time])

mh_recalls = filter(c -> c.sampler == "mh_sampler", recalls)
mh_times = filter(c -> c.sampler == "mh_sampler", times)

mtm_recalls = filter(c -> c.sampler == "mtm_sampler", recalls)
mtm_times = filter(c -> c.sampler == "mtm_sampler", times)
recall_groups = groupby(mtm_recalls, :n_particles)
time_groups = groupby(mtm_times, :n_particles)

# Actually plot it
fig = MK.Figure(resolution=(DISS_WIDTH, 2 / 3 * DISS_WIDTH); figure_padding=10)
ax_vsd = MK.Axis(fig[2, 1]; xlabel="pose inference time / s", ylabel="recall", title="VSD", limits=(nothing, (0, 1)), yticks=0:0.25:1)
for (rec, tim) in zip(recall_groups, time_groups)
    MK.lines!(ax_vsd, tim.mean_time, rec.vsd_recall; label="MTM $(rec.n_particles |> first) particles")
end
MK.lines!(ax_vsd, mh_times.mean_time, mh_recalls.vsd_recall; label="MCMC-MH")
MK.vlines!(ax_vsd, [0.5]; color=:black, linestyle=:dash)

ax_adds = MK.Axis(fig[2, 2]; xlabel="pose inference time / s", ylabel="recall", title="ADDS", limits=(nothing, (0, 1)), yticks=0:0.25:1)
for (rec, tim) in zip(recall_groups, time_groups)
    MK.lines!(ax_adds, tim.mean_time, rec.adds_recall; label="MTM $(rec.n_particles |> first) particles")

end
MK.lines!(ax_adds, mh_times.mean_time, mh_recalls.adds_recall; label="MCMC-MH")
MK.vlines!(ax_adds, [0.5]; color=:black, linestyle=:dash)

ga = fig[1, :] = MK.GridLayout()
ax_vsdbop = MK.Axis(ga[1, 1]; xlabel="pose inference time / s", ylabel="recall", title="VSDBOP", limits=(nothing, (0, 1)), yticks=0:0.25:1)
for (rec, tim) in zip(recall_groups, time_groups)
    MK.lines!(ax_vsdbop, tim.mean_time, rec.vsdbop_recall; label="MTM $(rec.n_particles |> first) particles")
end
MK.lines!(ax_vsdbop, mh_times.mean_time, mh_recalls.vsdbop_recall; label="MCMC-MH")
MK.vlines!(ax_vsdbop, [0.5]; color=:black, linestyle=:dash)
MK.Legend(ga[1, 2], ax_vsdbop)

# display(fig)
save(joinpath("plots", "$experiment_name.pdf"), fig)
