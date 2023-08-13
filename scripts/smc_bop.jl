# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# TODO create my own smaller training dataset using BlenderProc - the datasets from the website are too large

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

CUDA.allowscalar(false)

# Load the depth image, mask image, and object mesh
function load_img_mesh(df_row, parameters, gl_context)
    depth_img = load_depth_image(df_row, parameters.img_size...) |> device_array_type(parameters)
    mask_img = load_mask_image(df_row, parameters.img_size...) |> device_array_type(parameters)
    mesh = upload_mesh(gl_context, load_mesh(df_row))
    depth_img, mask_img, mesh
end

# Report the wall time from the point right after the raw data (the image, 3D object models etc.) is loaded
function timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row, sampler)
    # Context
    cpu_rng = Random.default_rng(parameters)
    dev_rng = device_rng(parameters)

    time = @elapsed begin
        # Setup experiment
        camera = crop_camera(df_row)

        prior_o = fill(parameters.float_type(parameters.o_mask_not), parameters.width, parameters.height) |> device_array_type(parameters)
        # NOTE Result / conclusion: adding masks makes the algorithm more robust and allows higher σ_t (quantitative difference of how much offset in the prior_t is possible?)
        prior_o[mask_img] .= parameters.o_mask_is

        prior_t = point_from_segmentation(df_row.bbox, depth_img, mask_img, df_row.cv_camera)
        # TODO If using point prior: bias position prior by a fixed distance: recall / bias curve
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

# Save results per scene via DrWatson's produce_or_load
function scene_inference(config)
    # Extract config and load dataset
    @unpack scene_id, dataset, testset, sampler = config
    sampler = eval(sampler)
    bop_full_path = datadir("bop", dataset, testset)
    if occursin("test", testset)
        scene_df = test_targets(bop_full_path, scene_id)
    elseif occursin("train", testset) || occursin("val", testset)
        scene_df = train_targets(bop_full_path, scene_id)
    end

    # Setup parameters
    parameters = Parameters()
    @reset parameters.n_steps = 200
    @reset parameters.n_particles = 100

    # Store result in DataFrame
    result_df = select(scene_df, :scene_id, :img_id, :obj_id)
    result_df.score = Vector{parameters.float_type}(undef, nrow(result_df))
    result_df.R = Vector{Quaternion{parameters.float_type}}(undef, nrow(result_df))
    result_df.t = Vector{Vector{parameters.float_type}}(undef, nrow(result_df))
    result_df.time = Vector{Float64}(undef, nrow(result_df))
    result_df.final_state = Vector{SmcState}(undef, nrow(result_df))
    result_df.states = Vector{Vector{SmcState}}(undef, nrow(result_df))

    # Make sure the context is destroyed to avoid undefined behavior
    gl_context = render_context(parameters)
    try
        # Avoid timing the pre-compilation
        df_row = first(scene_df)
        depth_img, mask_img, mesh = load_img_mesh(df_row, parameters, gl_context)
        timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row, sampler)

        # Run inference per detection
        for (idx, df_row) in enumerate(eachrow(scene_df))
            # Image crops differ per object
            depth_img, mask_img, mesh = load_img_mesh(df_row, parameters, gl_context)
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
        Dict("parameters" => parameters, "results" => result_df)
    finally
        destroy_context(gl_context)
    end
end

bop_datasets = [("lmo", "test"), ("tless", "test_primesense"), ("itodd", "val")]
@progress for bop_dataset in bop_datasets
    # DrWatson configuration
    dataset, testset = bop_dataset
    bop_full_path = datadir("bop", bop_dataset...)
    scene_id = bop_scene_ids(bop_full_path)
    sampler = [:smc_mh, :smc_forward]
    config = @dict dataset testset scene_id sampler
    dicts = dict_list(config)

    # Run and save results
    result_path = datadir("exp_raw", "baseline")
    for d in dicts
        produce_or_load(scene_inference, d, result_path; filename=c -> savename(c; connector=","))
    end
end
