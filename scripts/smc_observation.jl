# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

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
experiment_name = "smc_observation"
result_dir = datadir("exp_raw", experiment_name)
dataset = ["lm"] # TODO , "tless", "itodd"]
pixel = [:no_exp, :exp, :smooth]
# Classification and regularization: 
# :no - no classification, simple regularization
# :simple - classification, simple regularization
# :class - classification, L0 regularization
classification = [:no, :simple, :class]
testset = "train_pbr"
scene_id = 0 # TODO [0:4...]
configs = dict_list(@dict dataset testset scene_id pixel classification)

"""

Returns `pixel_fn(μ, o)` and `class_fn(prior_o, μ, z)`
"""
function pixel_model(pixel, parameters)
    # force evaluating parameters before capturing in closure
    σ = parameters.pixel_σ
    σ_class = parameters.association_σ
    θ = parameters.pixel_θ
    min_depth = parameters.min_depth
    max_depth = parameters.max_depth
    if pixel == :no_exp
        pixel_fn = (μ, o) -> begin
            normal = KernelNormal(μ, σ)
            tail = TailUniform(min_depth, max_depth)
            BinaryMixture(normal, tail, o, one(o) - o)
        end
        class_fn = (o, μ, z) -> begin
            # TODO why condition inside marginalized_association?
            dist_is = x -> KernelNormal(x, σ)
            dist_not = _ -> TailUniform(min_depth, max_depth)
            marginalized_association(dist_is, dist_not, o, μ, z)
        end
        return (pixel_fn, class_fn)
    elseif pixel == :exp
        pixel_fn = (μ, o) -> begin
            normal = KernelNormal(μ, σ)
            tail = pixel_tail(min_depth, max_depth, θ, σ, μ)
            BinaryMixture(normal, tail, o, one(o) - o)
        end
        class_fn = (o, μ, z) -> begin
            dist_is = x -> KernelNormal(x, σ_class)
            dist_not = x -> pixel_tail(min_depth, max_depth, θ, σ_class, x)
            marginalized_association(dist_is, dist_not, o, μ, z)
        end
        return (pixel_fn, class_fn)
    elseif pixel == :smooth
        pixel_fn = (μ, o) -> begin
            normal = KernelNormal(μ, σ)
            tail = smooth_tail(min_depth, max_depth, θ, σ, μ)
            BinaryMixture(normal, tail, o, one(o) - o)
        end
        class_fn = (o, μ, z) -> begin
            dist_is = x -> KernelNormal(x, σ_class)
            dist_not = x -> smooth_tail(min_depth, max_depth, θ, σ_class, x)
            marginalized_association(dist_is, dist_not, o, μ, z)
        end
        return (pixel_fn, class_fn)
    end
end

"""
    rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row)
Assembles the posterior model and the sampler from the loaded images, mesh, and DataFrame row.
"""
function rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row, pixel, classification)
    # Context
    cpu_rng = Random.default_rng(parameters)
    dev_rng = device_rng(parameters)

    # Setup experiment
    camera = crop_camera(df_row)
    # TODO use from row?
    prior_o = fill(parameters.float_type(parameters.o_mask_not), parameters.width, parameters.height)
    prior_o[mask_img] .= parameters.o_mask_is
    # Prior t from mask is imprecise no need to bias
    prior_t = point_from_segmentation(df_row.bbox, depth_img, mask_img, df_row.cv_camera)
    experiment = Experiment(gl_context, Scene(camera, [mesh]), prior_o, prior_t, depth_img)

    # Model
    μ_node = point_prior(parameters, experiment, cpu_rng)
    pixel_fn, class_fn = pixel_model(pixel, parameters)
    # TODO this is the special sauce
    if classification == :no
        o_node = BroadcastedNode(:o, dev_rng, KernelDirac, experiment.prior_o)
    else
        o_node = DeterministicNode(:o, μ -> class_fn.(experiment.prior_o, μ, experiment.depth_image), (μ_node,))
    end
    z = BroadcastedNode(:z, dev_rng, pixel_fn, (μ_node, o_node))
    if classification == :class
        z_norm = ModifierNode(z, dev_rng, SimpleImageRegularization | parameters.c_reg)
    else
        z_norm = ModifierNode(z, dev_rng, SimpleImageRegularization | parameters.c_reg)
    end
    posterior = PosteriorModel(z_norm | experiment.depth_image)

    # Sampler
    sampler = smc_mh(cpu_rng, parameters, posterior)
    # Result
    cpu_rng, posterior, sampler
end

"""
    timed_inference
Report the wall time from the point right after the raw data (the image, 3D object models etc.) is loaded.
"""
function timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row, pixel, classification)
    timed = @elapsed begin
        rng, posterior, sampler = rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row, pixel, classification)

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
    @unpack dataset, testset, scene_id, pixel, classification = config
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
    depth_img, mask_img, mesh = load_img_mesh(df_row, parameters, gl_context)
    rng, posterior, sampler = rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row, pixel, classification)
    step_time = mean_step_time(rng, posterior, sampler)
    # time budget of 0.5 seconds
    @reset parameters.n_steps = floor(Int, parameters.time_budget / step_time)

    # Run inference per detection
    @progress "dataset: $(dataset), testset: $(testset), scene_id : $(scene_id)" for (idx, df_row) in enumerate(eachrow(scene_df))
        # Image crops differ per object
        depth_img, mask_img, mesh = load_img_mesh(df_row, parameters, gl_context)
        # Run and collect results
        t, R, score, timed = timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row, pixel, classification)
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
@progress "SMC observation model" for config in configs
    @produce_or_load(gl_scene_inference, config, result_dir; filename=c -> savename(c; connector=","))
end
destroy_context(gl_context)

# Evaluate
evaluate_errors(experiment_name)
function parse_config(path)
    config = my_parse_savename(path)
    @unpack pixel, classification, dataset = config
    pixel, classification, dataset
end

# Calculate recalls
pro_df = collect_results(datadir("exp_pro", experiment_name, "errors"))
transform!(pro_df, :path => ByRow(parse_config) => [:pixel, :classification, :dataset])
# Threshold errors
transform!(pro_df, :adds => ByRow(x -> threshold_errors(x, ADDS_θ)) => :adds_thresh)
transform!(pro_df, :vsd => ByRow(x -> threshold_errors(x, BOP18_θ)) => :vsd_thresh)
transform!(pro_df, :vsdbop => ByRow(x -> threshold_errors(vcat(x...), BOP19_THRESHOLDS)) => :vsdbop_thresh)

# Recall by pixel model and classification
groups = groupby(pro_df, [:pixel, :classification])
recalls = combine(groups, :adds_thresh => (x -> recall(x...)) => :adds_recall, :vsd_thresh => (x -> recall(x...)) => :vsd_recall, :vsdbop_thresh => (x -> recall(x...)) => :vsdbop_recall)
CSV.write(datadir("exp_pro", experiment_name, "pixel_classification_recall.csv"), recalls)
display(recalls)

# TODO plot heatmap of components
