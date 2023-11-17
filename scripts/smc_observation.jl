# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

"""
Different observation models for "Modeling Occlusions: Exponential Distribution,
Classification, Regularization"
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
using Printf
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
dataset = ["itodd", "lm", "tless"]
pixel = [:no_exp, :exp, :smooth]
o_prior = [:flat, :mask]
# Classification and regularization:
# :no - no classification, simple regularization
# :simple - classification, simple regularization
# :class - classification, L0 regularization
# TODO it seems like a lower pixel_σ is required to make the class work
classification = [:no, :simple, :class]
testset = "train_pbr"
scene_id = [0:4...]
configs = dict_list(@dict dataset testset scene_id pixel classification o_prior)

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
            # TODO simplify interface: why condition inside marginalized_association?
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
    rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row, config)
Assembles the posterior model and the sampler from the loaded images, mesh, and DataFrame row.
"""
function rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row, config)
    @unpack classification, pixel, o_prior = config
    # Context
    cpu_rng = Random.default_rng(parameters)
    dev_rng = device_rng(parameters)

    # Setup experiment
    camera = crop_camera(df_row)
    if o_prior == :flat
        prior_o = parameters.float_type(0.5)
    elseif o_prior == :mask
        prior_o = fill(parameters.float_type(parameters.o_mask_not), parameters.width, parameters.height)
        prior_o[mask_img] .= parameters.o_mask_is
    end
    # Prior t from mask is imprecise no need to bias
    prior_t = point_from_segmentation(df_row.bbox, depth_img, mask_img, df_row.cv_camera)
    experiment = Experiment(gl_context, Scene(camera, [mesh]), prior_o, prior_t, depth_img)

    # Model
    μ_node = point_prior(parameters, experiment, cpu_rng)
    pixel_fn, class_fn = pixel_model(pixel, parameters)
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
function timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row, config)
    timed = @elapsed begin
        rng, posterior, sampler = rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row, config)

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
    @unpack dataset, testset, scene_id = config
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
    rng, posterior, sampler = rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row, config)
    step_time = mean_step_time(rng, posterior, sampler)
    @reset parameters.n_steps = floor(Int, parameters.time_budget / step_time)

    # Run inference per detection
    @progress "dataset: $(dataset), testset: $(testset), scene_id : $(scene_id)" for (idx, df_row) in enumerate(eachrow(scene_df))
        # Image crops differ per object
        depth_img, mask_img, mesh = load_img_mesh(df_row, parameters, gl_context)
        # Run and collect results
        t, R, score, timed = timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row, config)
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
    @unpack pixel, classification, dataset, o_prior = config
    pixel, classification, o_prior, dataset
end
# Pretty print experiment names, "smooth" and "exp" are fine
pixel_label(s) = s == "no_exp" ? "no" : s
function classification_label(s)
    if s == "no"
        "no, Lₚₓ"
    elseif s == "simple"
        "yes, Lₚₓ"
    elseif s == "class"
        "yes, L₀"
    end
end

# Calculate recalls
pro_df = collect_results(datadir("exp_pro", experiment_name, "errors"))
transform!(pro_df, :path => ByRow(parse_config) => [:pixel, :classification, :o_prior, :dataset])
# Threshold errors
transform!(pro_df, :adds => ByRow(x -> threshold_errors(x, ADDS_θ)) => :adds_thresh)
transform!(pro_df, :vsd => ByRow(x -> threshold_errors(x, BOP18_θ)) => :vsd_thresh)
transform!(pro_df, :vsdbop => ByRow(x -> threshold_errors(vcat(x...), BOP19_THRESHOLDS)) => :vsdbop_thresh)

# Recall by pixel model and classification
groups = groupby(pro_df, [:pixel, :classification, :o_prior])
recalls = combine(groups, :adds_thresh => (x -> recall(x...)) => :adds_recall, :vsd_thresh => (x -> recall(x...)) => :vsd_recall, :vsdbop_thresh => (x -> recall(x...)) => :vsdbop_recall)
# calculate core recall
recalls = transform(recalls, [:adds_recall, :vsd_recall, :vsdbop_recall] => ByRow((a, b, c) -> mean([a, b, c])) => :avg_recall)

CSV.write(datadir("exp_pro", experiment_name, "pixel_classification_recall.csv"), recalls)
display(recalls)

fig = MK.Figure(resolution=(DISS_WIDTH, 0.4 * DISS_WIDTH))
crange = (minimum(recalls.vsdbop_recall), maximum(recalls.vsdbop_recall))
# Heatmap for table
for (idx, group) in enumerate(groupby(recalls, :o_prior))
    # into matrix shape
    vsd_df = unstack(group, :classification, :pixel, :vsdbop_recall)
    # increasing complexity
    select!(vsd_df, [:classification, :no_exp, :exp, :smooth])
    permute!(vsd_df, [2, 3, 1])
    data = Array(vsd_df[:, 2:end])
    # title
    prior_type = first(group.o_prior)
    if prior_type == "mask"
        title = "Mask prior"
    elseif prior_type == "flat"
        title = "Flat prior"
    end
    # labeling
    column_names = pixel_label.(names(vsd_df, 2:4))
    row_names = classification_label.(vsd_df.classification)
    xticks = (eachindex(column_names), column_names)
    yticks = (eachindex(row_names), row_names)

    # Plot
    ax = MK.Axis(fig[1, idx]; title=title, xticks=xticks, yticks=yticks, xlabel="occlusion model", ylabel="class. & regul.", aspect=1)
    hm = MK.heatmap!(ax, data'; colorrange=crange)
    data_string(data) = @sprintf("%.3f", data)
    MK.text!(ax,
        data_string.(vec(data)),
        position=[MK.Point2f(x, y) for x in eachindex(column_names) for y in eachindex(row_names)],
        align=(:center, :center)
    )
    MK.colsize!(fig.layout, idx, MK.Aspect(1, 1.0))
end
MK.Colorbar(fig[1, 3]; label="recall / -", limits=crange)

# display(fig)
save(joinpath("plots", "$(experiment_name).pdf"), fig)
