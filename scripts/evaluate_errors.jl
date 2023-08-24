# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
Load the results from disk and evaluate pose error metrics.
"""

using DrWatson
@quickactivate("MCMCDepth")

using DataFrames
using MCMCDepth
using PoseErrors
using SciGL
using ThreadsX

using ProgressLogging
using TerminalLoggers
using Logging: global_logger
global_logger(TerminalLogger(right_justify=120))

"""
    load_df(directory, file)
Loads the experiments result as a DataFrame.
"""
function load_df(directory, file)
    dict = load(joinpath(directory, file))
    experiment_df = dict["result_df"]
    # Keep only relevant columns
    experiment_df = experiment_df[!, [:scene_id, :img_id, :obj_id, :t, :R, :score, :time]]

    # Load corresponding gt data
    _, config = parse_savename(file; connector=",")
    @unpack dataset, testset, scene_id = config
    gt_df = bop_test_or_train(dataset, testset, scene_id)

    # Add gt_R & gt_t for testset
    datasubset_path = datadir("bop", dataset, testset)
    if !("gt_t" in names(gt_df))
        leftjoin!(gt_df, PoseErrors.gt_dataframe(datasubset_path, scene_id)
            ; on=[:scene_id, :img_id, :gt_id])
    end
    if !("visib_fract" in names(gt_df))
        leftjoin!(gt_df, PoseErrors.gt_info_dataframe(datasubset_path, scene_id); on=[:scene_id, :img_id, :gt_id])
    end

    # Match estimates to ground truths
    joined = outerjoin(gt_df, experiment_df; on=[:scene_id, :img_id, :obj_id])
    # Ignore visib_fract < 0.1
    filter!(row -> row.visib_fract >= 0.1, joined)
    # Estimates without ground truth are not relevant for recall
    filter!(:gt_t => (x -> !ismissing(x)), joined)
end

# ADDS
es_pose(df_row) = to_pose(df_row.t, df_row.R)
gt_pose(df_row) = to_pose(df_row.gt_t, df_row.gt_R)

function adds_row(row)
    if ismissing(row.t)
        # No prediction -> always wrong -> ∞ error
        Inf32
    else
        mesh_eval = load_mesh_eval(row)
        points = mesh_eval.position
        gt = gt_pose(row)
        es = es_pose(row)
        normalized_adds_error(points, es, gt, row.diameter)
    end
end

# VSD
function vsd_row(row, dist_context, δ)
    if ismissing(row.t)
        # No prediction -> always wrong -> ∞ error
        Inf32
    else
        mesh = load_mesh_eval(row)
        gt = gt_pose(row)
        es = es_pose(row)
        cv_camera = row.cv_camera
        width, height = size(dist_context.render_data)
        depth_img = load_depth_image(row, width, height)
        dist_img = depth_to_distance(depth_img, cv_camera)
        # BOP19 and later use normalized version with multiple τ
        vsd_error(dist_context, cv_camera, mesh, dist_img, es, gt; δ=δ)
    end
end

# VSDBOP
function vsdbop_row(row, dist_context, δ)
    if ismissing(row.t)
        # No prediction -> always wrong -> ∞ error
        return fill(Inf32, length(BOP19_THRESHOLDS))
    else
        mesh = load_mesh_eval(row)
        gt = gt_pose(row)
        es = es_pose(row)
        cv_camera = row.cv_camera
        width, height = size(dist_context.render_data)
        depth_img = load_depth_image(row, width, height)
        dist_img = depth_to_distance(depth_img, cv_camera)
        # BOP19 and later use normalized version with multiple τ
        normalized_vsd_error(dist_context, cv_camera, mesh, dist_img, es, gt, row.diameter; δ=δ, τ=BOP19_THRESHOLDS)
    end
end

"""
    combine_estimates(group)
`group` must be grouped by the estimates, e.g via t.
Combines the errors per estimate by generating vectors sorted by the gt values.
Moreover, the score for the estimate is stored.
"""
function combine_estimates(group)
    # Order of ground truth must be the same
    sorted = sort(group, :gt_t)
    (adds=[sorted.adds], vsd=[sorted.vsd], vsdbop=[sorted.vsdbop], score=[sorted.score], gt_t=[sorted.gt_t])
end

"""
    match_obj_errors(group)
`group` must be grouped by the scene_id, image_id, and obj_id.
For this estimate, 
Greedily matches at most one ground truth to each estimated pose of the object.
If no estimate is available for a ground truth, Inf set as error.
"""
function match_obj_errors(group)
    estimate_groups = groupby(group, :t)
    errors_per_estimate = combine(estimate_groups, combine_estimates)
    score_per_estimate = first.(errors_per_estimate.score)
    # Returns Inf for each gt where no estimate is available.
    adds = match_errors(score_per_estimate, errors_per_estimate.adds)
    vsd = match_errors(score_per_estimate, errors_per_estimate.vsd)
    # TODO for each score in the arrays [[[for each θ in BOP19_THRESHOLDS] for each]]
    vsdbop = match_bop19_errors(score_per_estimate, errors_per_estimate.vsdbop)
    @ntuple adds vsd vsdbop
end

"""
    scene_errors(dist_context, scene_df, config)
Calculate and match the errors for each ground truth - estimate combination (per scene_id, img_id, obj_id).
"""
function calc_match_errors(dist_context, experiment_name, config)
    @unpack dataset = config
    # Load the estimates for the scene
    directory = datadir("exp_raw", experiment_name)
    file = savename(config, "jld2"; connector=",")
    df = load_df(directory, file)

    # Calculate different error metrics
    # Different VSD δ for visible surface in ITODD & steri
    vsd_δ = contains(dataset, "itodd") || contains(dataset, "steri") ? 5e-3 : BOP_δ |> Float32
    # WARN do not parallelize using ThreadsX, OpenGL is sequential
    df.vsd = map(row -> vsd_row(row, dist_context, vsd_δ), eachrow(df))
    df.vsdbop = map(row -> vsdbop_row(row, dist_context, vsd_δ), eachrow(df))
    # This is ~10x faster using ThreadsX
    df.adds = ThreadsX.map(adds_row, eachrow(df))

    # Greedy matching of the ground truth to the estimates
    errors_per_obj = groupby(df, [:scene_id, :img_id, :obj_id])
    matched = combine(match_obj_errors, errors_per_obj)
    # must be the number of ground truth annotations
    @assert nrow(matched) == nrow(unique(df, :gt_t))

    Dict("vsd" => matched.vsd, "vsdbop" => matched.vsdbop, "adds" => matched.adds)
end

# function evaluate_config(parameters, dist_context, directory, file, config)
#     experiment_df = load_df(directory, file)
#     errors = calc_match_errors(dist_context, experiment_df, config)

#     result_df = experiment_df[!, [:scene_id, :img_id, :obj_id, :gt_R, :gt_t, :R, :t, :score, :time, :visib_fract]]
#     result_df.adds = errors.adds
#     result_df.vsd = errors.vsd
#     result_df.vsdbop = errors.vsdbop

#     @strdict parameters result_df
# end

experiment_name = "baseline"
# Collect the results per experiment / scene
directory = datadir("exp_raw", experiment_name)
files = readdir(directory)
parameters = load(joinpath(directory, first(files)))["parameters"]
# TODO The size of the context does matter because of scaling & interpolation of the crop
dist_context = distance_offscreen_context(parameters.width, parameters.height, parameters.depth)

@progress "evaluating exp_raw/$experiment_name" for file in files
    _, config = parse_savename(file; connector=",")
    calc_match_closure(config) = calc_match_errors(dist_context, experiment_name, config)
    @produce_or_load(calc_match_closure, config, datadir("exp_pro", experiment_name, "errors"); filename=c -> savename(c; connector=","))
end

destroy_context(dist_context)
