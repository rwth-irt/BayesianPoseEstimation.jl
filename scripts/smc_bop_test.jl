# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
Run the algorithms on validation and test sets using ground truth masks
"""

using DrWatson
@quickactivate("MCMCDepth")

using Accessors
using CUDA
using CSV
using DataFrames
using MCMCDepth
using PoseErrors
using Random
using Rotations
using SciGL
using Statistics

using Logging
using ProgressLogging
using TerminalLoggers
global_logger(TerminalLogger(right_justify=120))

CUDA.allowscalar(false)

# General experiment
time_budget = 1
experiment_name = "smc_bop_test_$(time_budget)s"
result_dir = datadir("exp_raw", experiment_name)
parameters = Parameters()
@reset parameters.n_particles = 100
@reset parameters.depth = parameters.n_particles
@reset parameters.time_budget = time_budget

@reset parameters.o_mask_is = 0.9
@reset parameters.o_mask_not = 1 - parameters.o_mask_not
@reset parameters.pixel_σ = 0.005
@reset parameters.proposal_σ_r = fill(π, 3)

sampler = :smc_mh

dataset = "hb"
testset = "test_primesense"
scene_id = [3, 5, 13]
hb_config = dict_list(@dict sampler dataset testset scene_id)

dataset = "icbin"
testset = "test"
scene_id = [1:3...]
icbin_config = dict_list(@dict sampler dataset testset scene_id)

dataset = "itodd"
testset = "test"
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

dataset = "tudl"
testset = "test"
scene_id = [1:3...]
tudl_config = dict_list(@dict sampler dataset testset scene_id)

dataset = "ycbv"
testset = "test"
scene_id = [48:59...]
ycbv_config = dict_list(@dict sampler dataset testset scene_id)

configs = [hb_config..., icbin_config..., itodd_config..., lmo_config..., tless_config..., tudl_config..., ycbv_config...]

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
    scene_df = test_targets(datadir("bop", dataset, testset), scene_id; detections_file="default_detections_task4.json")

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
    mask_img = load_segmentation(df_row, parameters.img_size...) |> device_array_type(parameters)
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
        mask_img = load_segmentation(df_row, parameters.img_size...) |> device_array_type(parameters)
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
@progress "SMC BOP test" for config in configs
    @produce_or_load(gl_scene_inference, config, result_dir; filename=my_savename)
end
destroy_context(gl_context)

# Export BOP CSVs
# scene_id, im_id, obj_id, score, R, t, time
outdir = mkpath(datadir("exp_pro", experiment_name))
est_df = collect_results(datadir("exp_raw", experiment_name))
function parse_config(path)
    config = my_parse_savename(path)
    @unpack dataset, scene_id = config
    dataset, scene_id
end
DataFrames.transform!(est_df, :path => ByRow(parse_config) => [:dataset, :scene_id])

groups = groupby(est_df, :dataset)
for (key, group) in zip(keys(groups), groups)
    csv_df = DataFrame(scene_id=Int[], im_id=Int[], obj_id=Int[], score=Float64[], R=String[], t=String[], time=Float64[])
    for scene_row in eachrow(group)
        # all times must be the same for a scene and img - not true for my method so use mean
        mean_time = mean(scene_row.result_df.time)
        for row in eachrow(scene_row.result_df)
            # r_ij for i-th row and j-th column, separated by spaces
            R = row.R |> QuatRotation |> RotMatrix
            r = [R[i, j] for i in 1:size(R)[1] for j in 1:size(R)[2]]
            r_str = reduce(r) do x, y
                string(x) * " " * string(y)
            end
            # in mm, separated by spaces
            t = row.t .* 1e3
            t_str = reduce(t) do x, y
                string(x) * " " * string(y)
            end
            push!(csv_df, (; scene_id=row.scene_id, im_id=row.img_id, obj_id=row.obj_id, score=row.score, R=r_str, t=t_str, time=mean_time))
        end
    end
    CSV.write(joinpath(outdir, "$(key.dataset).csv"), csv_df)
end
