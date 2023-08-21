# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 


"""
Load the results from disk, evaluate pose error metrics, and calculate the average recall.
"""

using DrWatson
@quickactivate("MCMCDepth")

using CUDA
using DataFrames
using MCMCDepth
using PoseErrors
using SciGL
using ThreadsX

using ProgressLogging
using TerminalLoggers
using Logging: global_logger
global_logger(TerminalLogger(right_justify=120))

# Collect the results per experiment / scene
experiment_dir = datadir("exp_raw", "baseline")
# TODO loop over files with produce_or_load
files = readdir(experiment_dir)

# Inference results
experiment_file = files[70]
experiment_dict = load(joinpath(experiment_dir, experiment_file))
experiment_df = experiment_dict["results"]
# Keep only relevant columns
experiment_df = experiment_df[!, [:scene_id, :img_id, :obj_id, :t, :R, :score, :time]]

# Load corresponding gt data
_, config = parse_savename(experiment_file; connector=",")
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

# ADDS
es_pose(df_row) = to_pose(df_row.t, df_row.R)
gt_pose(df_row) = to_pose(df_row.gt_t, df_row.gt_R)

function adds_row(row)
    if ismissing(row.t)
        # No prediction -> always wrong -> ∞ error
        Inf32
    elseif ismissing(row.gt_t)
        # No ground truth -> ignore for recall
        missing
    else
        mesh_eval = load_mesh_eval(row)
        points = mesh_eval.position
        gt = gt_pose(row)
        es = es_pose(row)
        normalized_adds_error(points, es, gt, row.diameter)
    end
end

# TODO create only once
# TODO does the size of the context matter? Kind off - probably scaling & interpolation
WIDTH = HEIGHT = 100
DEPTH = 50
dist_context = distance_offscreen_context(WIDTH, HEIGHT, DEPTH, CuArray)
# different δ for ITODD & steri
vsd_δ = contains(config["dataset"], "itodd") || contains(config["dataset"], "steri") ? 5e-3 : BOP_δ |> Float32

# VSD
function vsd_row(row, dist_context, δ)
    if ismissing(row.t)
        # No prediction -> always wrong -> ∞ error
        Inf32
    elseif ismissing(row.gt_t)
        # No ground truth -> ignore for recall
        missing
    else
        mesh = load_mesh_eval(row)
        gt = gt_pose(row)
        es = es_pose(row)
        cv_camera = row.cv_camera
        depth_img = load_depth_image(row, WIDTH, HEIGHT)
        dist_img = depth_to_distance(depth_img, cv_camera) |> CuArray
        # BOP19 and later use normalized version with multiple τ
        vsd_error(dist_context, cv_camera, mesh, dist_img, es, gt; δ=δ)
    end
end
# WARN do not parallelize using ThreadsX, OpenGL is sequential. 


# TODO VSDBOP
function vsdbop_row(row, dist_context, δ)
    if ismissing(row.t)
        # No prediction -> always wrong -> ∞ error
        return fill(Inf32, length(BOP19_THRESHOLDS))
    elseif ismissing(row.gt_t)
        # No ground truth -> ignore for recall
        return missing
    else
        mesh = load_mesh_eval(row)
        gt = gt_pose(row)
        es = es_pose(row)
        cv_camera = row.cv_camera
        depth_img = load_depth_image(row, WIDTH, HEIGHT)
        dist_img = depth_to_distance(depth_img, cv_camera) |> CuArray
        # BOP19 and later use normalized version with multiple τ
        normalized_vsd_error(dist_context, cv_camera, mesh, dist_img, es, gt, row.diameter; δ=δ, τ=BOP19_THRESHOLDS)
    end
end

# append results
result_df = joined[!, [:scene_id, :img_id, :obj_id, :gt_R, :gt_t, :R, :t, :score, :time, :visib_fract]]
@withprogress name = "Error metrics" begin
    result_df.adds = ThreadsX.map(adds_row, eachrow(joined))
    @logprogress 1 // 3
    # WARN do not parallelize using ThreadsX, OpenGL is sequential
    result_df.vsd = map(row -> vsd_row(row, dist_context, vsd_δ), eachrow(joined))
    @logprogress 2 // 3
    result_df.vsd_bop = map(row -> vsdbop_row(row, dist_context, vsd_δ), eachrow(joined))
end;
# Filter out missing gt data, not relevant for recall
filter!(:adds => x -> !ismissing(x), result_df)

# TODO weird that mcmc methods are that bad - rerun experiments?
println(sum(result_df.vsd .< 0.3))
println(sum(result_df.adds .< 0.1))

destroy_context(dist_context)

# TODO remember when matching: multiple errors are reported for vsdbop due to multiple τ

# TODO match per object
errors_per_obj = groupby(result_df, [:scene_id, :img_id, :obj_id])

# TODO loop groups
group = last(errors_per_gt)
group.adds

# TODO I think this is wrong - am I actually evaluating [[error(est, gt) for est in estimates] for gt in annotations]?
errors_per_gt = groupby(group, :gt_t)
function combine_per_gt(group)
    sort!(group, :score)
    (adds=[group.adds], score=[group.score])
end
combined_errors = combine(errors_per_gt, combine_per_gt)
println(combined_errors.score)
println(combined_errors.adds)

match_errors(Float32.(first(combined_errors.score)), combined_errors.adds)

adds_errors = group.adds

first(groups).vsd_bop

# TODO is ADDS recall suspiciously better than VSD
