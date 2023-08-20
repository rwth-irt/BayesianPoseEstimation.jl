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

# Collect the results per experiment / scene
result_dir = datadir("exp_raw", "baseline")
files = readdir(result_dir)

# Inference results
# TODO loop
result_file = last(files)
result_dict = load(joinpath(result_dir, result_file))
result_df = result_dict["results"]

# Load corresponding gt data
_, result_config = parse_savename(result_file; connector=",")
@unpack dataset, testset, scene_id = result_config
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

# Keep only relevant columns
result_df = result_df[!, [:scene_id, :img_id, :obj_id, :t, :R, :score, :time]]
gt_df[!, [:scene_id, :img_id, :obj_id, :gt_R, :gt_t, :cv_camera, :mesh_eval_path, :visib_fract]]

# Calculate errors per ground truth
joined = outerjoin(gt_df, result_df; on=[:scene_id, :img_id, :obj_id])
groups = groupby(joined, [:scene_id, :img_id, :obj_id, :gt_t])

# TODO combine(groups, adds_per_gt => : adds, vsd_per_gt => vsd, vsdbop_per_gt => vsdbop)
group = first(groups)
mesh = group |> first |> load_mesh_eval

# ADDS
points = mesh.position
diameter = model_diameter(points)

# TODO loop rows
row = first(group)
es_pose = to_pose(row.t, row.R)
gt_pose = to_pose(row.gt_t, row.gt_R)
# TODO move normalization to PoseErrors - new method or named parameter?
normalized_error = adds_error(points, es_pose, gt_pose) / diameter

# VSD
# TODO create only once
# TODO does the size of the context matter?
WIDTH = HEIGHT = 200
DEPTH = 100
dist_context = distance_offscreen_context(WIDTH, HEIGHT, DEPTH, CuArray)
cv_camera = row.cv_camera
depth_img = load_depth_image(row, WIDTH, HEIGHT)
dist_img = depth_to_distance(depth_img, cv_camera) |> CuArray
# TODO change order of τ and δ to enable default δ
# BUG this is too high
vsd_error(dist_context, cv_camera, mesh, dist_img, est_pose, gt_pose) #, BOP_δ, PoseErrors.BOP_18_τ)

destroy_context(dist_context)

# TODO VSDBOP