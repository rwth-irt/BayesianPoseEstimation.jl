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

@info "Loading packages"
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
        sampler = mh_sampler(cpu_rng, parameters, posterior)
        chain = sample(cpu_rng, posterior, sampler, parameters.n_steps; discard_initial=parameters.n_burn_in, thinning=parameters.n_thinning, progress=false)

        # Extract best pose and score
        score, idx = findmax(loglikelihood.(chain))
        t = variables(chain[idx]).t
        r = variables(chain[idx]).r
    end
    t, r, score, time
end

"""
scene_inference(gl_context, config)
    Save results per scene via DrWatson's produce_or_load for the `config`
"""
function scene_inference(gl_context, config)
    # Extract config and load dataset
    @unpack dataset, testset, scene_id, n_steps = config
    parameters = Parameters()
    @reset parameters.n_steps = n_steps

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
    timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row)

    # Run inference per detection
    @progress "Sampling poses" for (idx, df_row) in enumerate(eachrow(result_df))
        # Image crops differ per object
        depth_img, mask_img, mesh = load_img_mesh(df_row, parameters, gl_context)
        # Run and collect results
        t, R, score, timed = timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row)
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

dataset = ["lm", "tless", "itodd"]
testset = "train_pbr"
scene_id = 0
n_steps = [50, 100, 200, 500, 750, 1_000, 2_000, 3_000, 5_000]
configs = dict_list(@dict dataset testset scene_id n_steps)
experiment_name = "recall_n_steps"
result_dir = datadir("exp_raw", experiment_name)
@progress "Inference n_steps MH" for config in configs
    @produce_or_load(gl_scene_inference, config, result_dir; filename=my_savename)
end

destroy_context(gl_context)

# Calculate errors
evaluate_errors(experiment_name)

# Combine results by n_steps & dataset
result_df = collect_results(datadir("exp_pro", experiment_name, "errors"))
function parse_config(path)
    config = my_parse_savename(path)
    @unpack n_steps, dataset = config
    n_steps, dataset
end
transform!(result_df, :path => ByRow(parse_config) => [:n_steps, :dataset])

# Threshold errors
transform!(result_df, :adds => ByRow(x -> threshold_errors(x, ADDS_θ)) => :adds_thresh)
transform!(result_df, :vsd => ByRow(x -> threshold_errors(x, BOP18_θ)) => :vsd_thresh)
transform!(result_df, :vsdbop => ByRow(x -> threshold_errors(vcat(x...), BOP19_THRESHOLDS)) => :vsdbop_thresh)

# Recall & time by n_steps
groups = groupby(result_df, :n_steps)
recalls = combine(groups, :adds_thresh => (x -> recall(x...)) => :adds_recall, :vsd_thresh => (x -> recall(x...)) => :vsd_recall, :vsdbop_thresh => (x -> recall(x...)) => :vsdbop_recall)

# Mean inference time
raw_df = collect_results(result_dir)
transform!(raw_df, :path => ByRow(parse_config) => [:n_steps, :dataset])
groups = groupby(raw_df, [:n_steps])
times = combine(groups, :time => (x -> mean(vcat(x...))) => :mean_time)

# Visualization in plot_mcmc_particles.jl
include("plot_mcmc_particles.jl")
