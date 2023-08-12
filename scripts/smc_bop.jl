# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using DrWatson
@quickactivate("MCMCDepth")

using Accessors
using BenchmarkTools
using CUDA
using DataFrames
using MCMCDepth
using PoseErrors
using Logging
using Random
using SciGL

using ProgressLogging
using TerminalLoggers

# Context
CUDA.allowscalar(false)
# Avoid timing the compilation
first_run = true
parameters = Parameters()
@reset parameters.n_steps = 200
@reset parameters.n_particles = 100
cpu_rng = Random.default_rng(parameters)
dev_rng = device_rng(parameters)
gl_context = render_context(parameters)

function load_img_mesh(df_row, parameters, gl_context)
    depth_img = load_depth_image(df_row, parameters.img_size...) |> device_array_type(parameters)
    mask_img = load_mask_image(df_row, parameters.img_size...) |> device_array_type(parameters)
    mesh = upload_mesh(gl_context, load_mesh(df_row))
    depth_img, mask_img, mesh
end

# Report the wall time from the point right after the raw data (the image, 3D object models etc.) is loaded
function timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row, sampler)
    time = @elapsed begin
        # Setup experiment
        camera = crop_camera(df_row)

        prior_o = fill(parameters.float_type(parameters.o_mask_not), parameters.width, parameters.height) |> device_array_type(parameters)
        # NOTE Result / conclusion: adding masks makes the algorithm more robust and allows higher σ_t (quantitative difference of how much offset in the prior_t is possible?)
        prior_o[mask_img] .= parameters.o_mask_is

        prior_t = point_from_segmentation(df_row.bbox, depth_img, mask_img, df_row.cv_camera)
        # TODO bias position prior by a fixed distance: recall / bias curve
        # pos_bias = parameters.bias_t * normalize(randn(cpu_rng, 3)) .|> parameters.float_type
        experiment = Experiment(gl_context, Scene(camera, [mesh]), prior_o, prior_t, depth_img)

        # Setup model
        prior = point_prior(parameters, experiment, cpu_rng)
        # posterior = association_posterior(parameters, experiment, prior, dev_rng)
        # NOTE no association → prior_o has strong influence
        posterior = simple_posterior(parameters, experiment, prior, dev_rng)
        # posterior = smooth_posterior(parameters, experiment, prior, dev_rng)

        sampler = sampler(cpu_rng, parameters, posterior)
        states, final_state = smc_inference(cpu_rng, posterior, sampler, parameters)

        # Extract best pose and score
        sample = final_state.sample
        score, idx = findmax(loglikelihood(sample))
        t = variables(sample).t[:, idx]
        r = variables(sample).r[idx]
    end
    t, r, score, final_state, states, time
end

# Save results per scene
# TODO wrap in function and call it DrWatson style as an experiment which allows to check whether it has been run before.
function scene_inference(config)
    # Extract config
    @unpack sceneid, dataset, sampler = config
    sampler = eval(sampler)
    bop_subset_dir = datadir("bop", dataset)

    # TODO run inference and save results on a per-scene basis
    scene_df = test_targets(bop_subset_dir, scene_id)

    # TODO generate: scene_df, parameters, gl_context
    result_df = select(scene_df, :scene_id, :img_id, :obj_id)
    result_df.score = Vector{parameters.float_type}(undef, nrow(result_df))
    result_df.R = Vector{Quaternion{parameters.float_type}}(undef, nrow(result_df))
    result_df.t = Vector{Vector{parameters.float_type}}(undef, nrow(result_df))
    result_df.time = Vector{Float64}(undef, nrow(result_df))
    result_df.final_state = Vector{SmcState}(undef, nrow(result_df))
    result_df.states = Vector{Vector{SmcState}}(undef, nrow(result_df))


    # Avoid timing the pre-compilation
    df_row = first(scene_df)
    depth_img, mask_img, mesh = load_img_mesh(df_row, parameters, gl_context)
    timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row, sampler)

    @progress for (idx, df_row) in enumerate(eachrow(scene_df))
        # Image crops differ per object
        depth_img = load_depth_image(df_row, parameters.img_size...) |> device_array_type(parameters)
        mask_img = load_mask_image(df_row, parameters.img_size...) |> device_array_type(parameters)
        mesh = upload_mesh(gl_context, load_mesh(df_row))
        # Run and collect results
        t, R, score, final_state, states, time = timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row, sampler)
        result_df[idx, :].score = score
        result_df[idx, :].R = R
        result_df[idx, :].t = t
        result_df[idx, :].time = time
        result_df[idx, :].final_state = final_state
        result_df[idx, :].states = states
    end
    # Return result
    result_df
end

# TODO iterate over scenes
# Dataset → DataFrames
bop_dataset = joinpath("tless", "test_primesense")
bop_subset_dir = datadir("bop", test_dir)
scene_ids = bop_scene_ids(bop_subset_dir)
config = Dict("dataset" => bop_dataset, "sceneid" => scene_ids, "sampler" => :smc_mh)
dicts = dict_list(config)

scene_inference(dicts[1])

produce_or_load(dicts[1])


# TODO incrementally write the dataframe to disk?
begin
    color_img = load_color_image(df_row, parameters.width, parameters.height)
    camera = crop_camera(df_row)
    @reset mesh.pose = to_pose(t, r)
    plot_scene_ontop(gl_context, Scene(camera, [mesh]), color_img)
end

# TODO DrWatson would save each run in a different file and then collect the results in a single DataFrame. However, they would run a simulation per parameter and not per data label.
result_data = Dict("parameters" => parameters, "data" => result_df)
result_root = datadir("exp_raw", bop_subset..., "smc")
@tagsave(datadir(result_root, "sim_1.jld2"), result_data)
# TODO how to read the tags? Probably a tag in the dictionary which is saved via JLD2

# TODO How to organize save-files? Per scene might be risky if something interrupts the computation. Per image seems reasonable.

# WARN when calculating the score for a dataset: test_targets does not contain all instances since some detections are missing. Use the gt_targets instead
gt_df = gt_targets(bop_subset_dir, 19)
