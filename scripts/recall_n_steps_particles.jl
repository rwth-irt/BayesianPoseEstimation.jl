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
using BenchmarkTools
using CUDA
using DataFrames
using MCMCDepth
using PoseErrors
using Logging
using Random
using SciGL
using ThreadsX

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
dataset = ["lm", "tless", "itodd"]
testset = "train_pbr"
scene_id = 0

# MTM
sampler = [:mtm_sampler]
n_particles = [5, 10, 20, 40]
# Maximum of 10_000 n_hypotheses (n_particles * n_steps) ~ 1.5 sec per inference
n_hypotheses = [500, 1_000, 1_500, 2_000, 3_000, 5_000, 10_000]
configs = dict_list(@dict dataset testset scene_id n_particles sampler)
result_dir = datadir("exp_raw", experiment_name)
@progress "MTM n_steps and n_particles" for config in configs
    @progress "MTM for $(config[:n_particles]) particles" for n_hyp in n_hypotheses
        config[:n_steps] = n_hyp ÷ config[:n_particles]
        @produce_or_load(gl_scene_inference, config, result_dir; filename=c -> savename(c; connector=","))
    end
end

# SMC
sampler = [:smc_bootstrap, :smc_forward, :smc_mh]
n_particles = [10, 25, 50, 100]
# Maximum of 25_000 n_hypotheses (n_particles * n_steps) ~ 1.5 sec per inference
n_hypotheses = [500, 1_000, 2_000, 3_000, 5_000, 15_000, 25_000]
configs = dict_list(@dict dataset testset scene_id n_particles sampler)
@progress "SMC n_steps and n_particles" for config in configs
    @progress "SMC for $(config[:n_particles]) particles" for n_hyp in n_hypotheses
        config[:n_steps] = n_hyp ÷ config[:n_particles]
        @produce_or_load(gl_scene_inference, config, result_dir; filename=c -> savename(c; connector=","))
    end
end

destroy_context(gl_context)

# Calculate errors
include("evaluate_errors.jl")

# Plot
using Plots
pythonplot()
diss_defaults()

function parse_config(path)
    _, config = parse_savename(path; connector=",")
    @unpack n_steps, n_particles, dataset, sampler = config
    n_steps, n_particles, dataset, sampler
end

all_pro = collect_results(datadir("exp_pro", experiment_name, "errors"))
transform!(all_pro, :path => ByRow(parse_config) => [:n_steps, :n_particles, :dataset, :sampler])

all_raw = collect_results(result_dir)
transform!(all_raw, :path => ByRow(parse_config) => [:n_steps, :n_particles, :dataset, :sampler])

# Load and filter data per sampler: 
for sampler_name in ["mtm_sampler", "smc_bootstrap", "smc_forward", "smc_mh"]
    pro_df = filter(x -> x.sampler == sampler_name, all_pro)
    raw_df = filter(x -> x.sampler == sampler_name, all_raw)

    # Threshold errors
    transform!(pro_df, :adds => ByRow(x -> threshold_errors(x, ADDS_θ)) => :adds_thresh)
    transform!(pro_df, :vsd => ByRow(x -> threshold_errors(x, BOP18_θ)) => :vsd_thresh)
    transform!(pro_df, :vsdbop => ByRow(x -> threshold_errors(vcat(x...), BOP19_THRESHOLDS)) => :vsdbop_thresh)

    # Recall & time by n_steps & n_particle
    groups = groupby(pro_df, [:n_steps, :n_particles])
    recalls = combine(groups, :adds_thresh => (x -> recall(x...)) => :adds_recall, :vsd_thresh => (x -> recall(x...)) => :vsd_recall, :vsdbop_thresh => (x -> recall(x...)) => :vsdbop_recall)

    # Mean inference time
    groups = groupby(raw_df, [:n_steps, :n_particles])
    times = combine(groups, :time => (x -> mean(vcat(x...))) => :mean_time)

    # Visualize per n_particles
    sort!(recalls, [:n_particles, :n_steps])
    sort!(times, [:n_particles, :n_steps])

    recall_groups = groupby(recalls, :n_particles)
    time_groups = groupby(times, :n_particles)

    # Lines
    MAX_TIME = 0.5
    p1 = plot()
    for (rec, tim) in zip(recall_groups, time_groups)
        plot!(tim.mean_time, rec.vsd_recall; label="$(rec.n_particles |> first) particles", xlabel="time / s", ylabel="VSD recall", ylims=[0, 1], legend=false)
    end
    vline!([MAX_TIME]; label=nothing, color=:black, linestyle=:dash)

    p2 = plot()
    for (rec, tim) in zip(recall_groups, time_groups)
        plot!(rec.n_steps * first(rec.n_particles), tim.mean_time; label="$(rec.n_particles |> first)", legend=:topleft, xlabel="hypotheses", ylabel="time / s")
    end
    hline!([MAX_TIME]; label=nothing, color=:black, linestyle=:dash)

    l = @layout [a; b]
    p = plot(p1, p2; layout=l)
    display(p)
    savefig(p, joinpath("plots", "recall_n_steps_particles_" * sampler_name * ".svg"))
end

# TODO Number of evaluated hypotheses / second does not grow linearly with the number of particles. Probably due to the overhead of starting the kernels. Plot this and then let each algorithm run for the same time [sec] since I only care about the compute TIME

# # Recall over n_steps & n_particles
# x = vcat([first(group.n_particles) for group in recall_groups]...)
# y = first(recall_groups).n_steps
# z_recall = hcat([group.vsd_recall for group in recall_groups]...)
# # string to avoid scaling
# h1 = heatmap(string.(x), string.(y), z_recall; xlabel="particles", ylabel="iterations", colorbar_title="VSD recall")

# # Normalized recall
# time_groups = groupby(times, :n_particles)
# z_time = hcat([group.mean_time for group in time_groups]...)
# z_norm = z_recall ./ z_time
# h1 = heatmap(string.(x), string.(y), z_norm; xlabel="particles", ylabel="iterations", colorbar_title="VSD recall / s")
