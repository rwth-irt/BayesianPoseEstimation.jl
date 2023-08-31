# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
Run different Metropolis Hastings MCMC on  the synthetic BOP datasets.
    Only the first scene of each dataset is evaluated because of the computation time.
    WARN: Results vary based on sampler configuration
    NOTE: Inference time grows linearly with n_hypotheses = n_particles * n_steps
    NOTE: smc_bootstrap & smc_forward mainly benefit from n_particles not n_steps
"""

using DrWatson
@quickactivate("MCMCDepth")

@info "Loading packages"
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

        # Sampling
        if sampler == :mtm_sampler
            mtm_sampler = eval(sampler)(cpu_rng, parameters, posterior)
            # Only MTM supports multiple particles. Metropolis Hastings in recall_n_steps.jl
            chain = sample(cpu_rng, posterior, mtm_sampler, parameters.n_steps; discard_initial=parameters.n_burn_in, thinning=parameters.n_thinning, progress=false)
            # Extract best pose and score
            score, idx = findmax(loglikelihood.(chain))
            t = variables(chain[idx]).t
            r = variables(chain[idx]).r
        else
            smc_sampler = eval(sampler)(cpu_rng, parameters, posterior)
            # Only MCMC or SMC style algorithms

            _, final_state = smc_inference(cpu_rng, posterior, smc_sampler, parameters)
            # Extract best pose and score
            final_sample = final_state.sample
            score, idx = findmax(loglikelihood(final_sample))
            t = variables(final_sample).t[:, idx]
            r = variables(final_sample).r[idx]
        end
    end
    t, r, score, time
end

"""
scene_inference(gl_context, config)
    Save results per scene via DrWatson's produce_or_load for the `config`
"""
function scene_inference(gl_context, config)
    # Extract config and load dataset
    @unpack dataset, testset, scene_id, n_steps, n_particles, sampler = config
    parameters = Parameters()
    @reset parameters.c_reg = 1 / 500
    @reset parameters.n_steps = n_steps
    @reset parameters.n_particles = n_particles

    result_df = bop_test_or_train(dataset, testset, scene_id)
    # Add gt_R & gt_t for testset
    datasubset_path = datadir("bop", dataset, testset)
    if !("gt_t" in names(result_df))
        leftjoin!(gt_df, PoseErrors.gt_dataframe(datasubset_path, scene_id)
            ; on=[:scene_id, :img_id, :gt_id])
    end
    if !("visib_fract" in names(result_df))
        leftjoin!(gt_df, PoseErrors.gt_info_dataframe(datasubset_path, scene_id); on=[:scene_id, :img_id, :gt_id])
    end

    # Store result in DataFrame. Numerical precision doesn't matter here → Float32
    result_df.score = Vector{Float32}(undef, nrow(result_df))
    result_df.R = Vector{Quaternion{Float32}}(undef, nrow(result_df))
    result_df.t = Vector{Vector{Float32}}(undef, nrow(result_df))
    time = Vector{Float32}(undef, nrow(result_df))

    # TODO I think this is where it fails since the sampler is assembled during timing which will measure some precompilation :/
    # Avoid timing the pre-compilation
    df_row = first(result_df)
    depth_img, mask_img, mesh = load_img_mesh(df_row, parameters, gl_context)
    timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row, sampler)

    # Run inference per detection
    @progress "Sampling poses" for (idx, df_row) in enumerate(eachrow(result_df))
        # Image crops differ per object
        depth_img, mask_img, mesh = load_img_mesh(df_row, parameters, gl_context)
        # Run and collect results
        t, R, score, timed = timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row, sampler)
        result_df[idx, :].score = score
        result_df[idx, :].R = R
        result_df[idx, :].t = t
        time[idx] = timed
    end
    @strdict parameters result_df time
end

gl_context = render_context(Parameters())
# Avoid recreating the context in scene_inference by conditioning on it / closure
gl_scene_inference = scene_inference | gl_context

experiment_name = "recall_n_steps_particles"
result_dir = datadir("exp_raw", experiment_name)
dataset = ["lm", "tless", "itodd"]
testset = "train_pbr"
scene_id = 0

# MTM
sampler = [:mtm_sampler]
n_particles = [5, 10, 20, 40, 250]
times = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.5, 3.0]
# Maximum of 10_000 n_hypotheses (n_particles * n_steps) ~ 1.5 sec per inference
n_hypotheses = [500, 1_000, 1_500, 2_000, 3_000, 5_000, 10_000]
configs = dict_list(@dict dataset testset scene_id n_particles sampler)
@progress "MTM n_steps and n_particles" for config in configs
    @progress "MTM for $(config[:n_particles]) particles" for tim in times
        config[:n_steps] = floor(Int, tim / step_time_100px(config[:sampler], config[:n_particles]))
        @produce_or_load(gl_scene_inference, config, result_dir; filename=c -> savename(c; connector=","))
    end
end

# SMC
sampler = [:smc_bootstrap, :smc_forward, :smc_mh]
n_particles = [10, 25, 50, 100, 250]
times = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
configs = dict_list(@dict dataset testset scene_id n_particles sampler)
@progress "SMC n_steps and n_particles" for config in configs
    @progress "SMC for $(config[:n_particles]) particles" for tim in times
        config[:n_steps] = floor(Int, tim / step_time_100px(config[:sampler], config[:n_particles]))
        @produce_or_load(gl_scene_inference, config, result_dir; filename=my_savename)
    end
end

destroy_context(gl_context)

# Calculate errors
evaluate_errors(experiment_name)

# Plot
using Plots
gr()
diss_defaults()

function parse_config(path)
    config = my_parse_savename(path)
    @unpack n_steps, n_particles, sampler = config
    n_steps, n_particles, sampler
end

# Calculate recalls
pro_df = collect_results(datadir("exp_pro", experiment_name, "errors"))
transform!(pro_df, :path => ByRow(parse_config) => [:n_steps, :n_particles, :sampler])
filter!(x -> x.n_particles > 1, pro_df)
# Threshold errors
transform!(pro_df, :adds => ByRow(x -> threshold_errors(x, ADDS_θ)) => :adds_thresh)
transform!(pro_df, :vsd => ByRow(x -> threshold_errors(x, BOP18_θ)) => :vsd_thresh)
transform!(pro_df, :vsdbop => ByRow(x -> threshold_errors(vcat(x...), BOP19_THRESHOLDS)) => :vsdbop_thresh)
groups = groupby(pro_df, [:sampler, :n_steps, :n_particles])
recalls = combine(groups, :adds_thresh => (x -> recall(x...)) => :adds_recall, :vsd_thresh => (x -> recall(x...)) => :vsd_recall, :vsdbop_thresh => (x -> recall(x...)) => :vsdbop_recall)

# Calculate mean pose inference times
raw_df = collect_results(result_dir)
transform!(raw_df, :path => ByRow(parse_config) => [:n_steps, :n_particles, :sampler])
filter!(x -> x.n_particles > 1, raw_df)
groups = groupby(raw_df, [:sampler, :n_steps, :n_particles])
times = combine(groups, :time => (x -> mean(vcat(x...))) => :mean_time)

# Actually plot it
function plot_sampler(sampler_name, recalls, times)
    recalls_filtered = filter(x -> x.sampler == sampler_name, recalls)
    times_filtered = filter(x -> x.sampler == sampler_name, times)
    sort!(recalls_filtered, [:n_particles, :n_steps])
    sort!(times_filtered, [:n_particles, :n_steps])
    # Visualize per n_particles
    recall_groups = groupby(recalls_filtered, :n_particles)
    time_groups = groupby(times_filtered, :n_particles)
    # Lines   
    p_adds = plot(; xlabel="pose inference time / s", ylabel="ADDS recall", ylims=[0, 1], linewidth=1.5)
    for (rec, tim) in zip(recall_groups, time_groups)
        plot!(p_adds, tim.mean_time, rec.adds_recall; legend=false)
    end
    vline!([0.5]; label=nothing, color=:black, linestyle=:dash, linewidth=1.5)

    p_vsd = plot(; xlabel="pose inference time / s", ylabel="VSD recall", ylims=[0, 1], linewidth=1.5)
    for (rec, tim) in zip(recall_groups, time_groups)
        plot!(p_vsd, tim.mean_time, rec.vsd_recall; legend=false)
    end
    vline!([0.5]; label=nothing, color=:black, linestyle=:dash, linewidth=1.5)

    p_vsdbop = plot(; xlabel="pose inference time / s", ylabel="VSDBOP recall", ylims=[0, 1], linewidth=1.5)
    for (rec, tim) in zip(recall_groups, time_groups)
        plot!(p_vsdbop, tim.mean_time, rec.vsdbop_recall; legend=:outerright, label="$(rec.n_particles |> first) particles")
    end
    vline!([0.5]; label=nothing, color=:black, linestyle=:dash, linewidth=1.5)

    lay = @layout [a; b c]
    p = plot(p_vsdbop, p_adds, p_vsd; layout=lay)
    display(p)
    savefig(p, joinpath("plots", "recall_n_steps_particles_" * sampler_name * ".pdf"))
end

for sampler_name in ["mtm_sampler", "smc_bootstrap", "smc_forward", "smc_mh"]
    plot_sampler(sampler_name, recalls, times)
end
