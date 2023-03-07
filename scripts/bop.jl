# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using FileIO
using MCMCDepth
using Plots
using SciGL

s_df = MCMCDepth.scene_dataframe("tless", "test_primesense", 1)
row = s_df[42, :]
width, height = row.img_size

parameters = Parameters()
# Context
@reset parameters.width = width
@reset parameters.height = height
@reset parameters.depth = 10

gl_context = render_context(parameters)

# Scene
camera = row.camera |> Camera
mesh = upload_mesh(gl_context, row.mesh)
# TODO convert all or nothing in BOP.jl? Mesh vs. Pose. Full scene probably not because of cropping
@reset mesh.pose = MCMCDepth.to_pose(row.cam_t_m2c, row.cam_R_m2c)
scene = Scene(camera, [mesh])

# Draw result for visual validation
MCMCDepth.diss_defaults()
gr()
rendered = draw(gl_context, scene)
# TODO load into DataFrame
img = load("/home/rd/code/mcmc-depth-images/datasets/tless/test_primesense/000001/rgb/000474.png")
MCMCDepth.plot_depth_ontop(img, rendered)

