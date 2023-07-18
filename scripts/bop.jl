# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using DataFrames
using FileIO
using ImageTransformations
using MCMCDepth
using Plots
using SciGL

# Datasets
s_df = scene_dataframe("tless", "test_primesense", 1)
@assert nrow(s_df) == 197
# s_df = scene_dataframe("itodd", "val", 1)
# s_df = scene_dataframe("lm", "test", 2)
row = s_df[100, :]

# Plot setup
MCMCDepth.diss_defaults()
gr()

# Context
parameters = Parameters()
@reset parameters.device = :CPU
@reset parameters.width = 400
@reset parameters.height = 400
@reset parameters.depth = 1
gl_context = render_context(parameters)

# Scene
camera = crop_camera(row)
mesh = upload_mesh(gl_context, row.mesh)
@reset mesh.pose = to_pose(row.cam_t_m2c, row.cam_R_m2c)
scene = Scene(camera, [mesh])

# Draw result for visual validation
color_img = load_color_image(row, parameters)
render_img = draw(gl_context, scene)
plot_depth_ontop(color_img, render_img, alpha=0.8)

mask_img = load_mask_image(row, parameters)

depth_img = load_depth_image(row, parameters)
plot_depth_img((depth_img .* mask_img))
plot_depth_img((render_img .* mask_img))

# TODO load camera noise depending on dataset name? Probabilistic Robotics: Larger Noise than expected? Tune parameter?
