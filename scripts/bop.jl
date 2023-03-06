# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using MCMCDepth
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
@reset parameters.cv_camera = row.camera
@reset parameters.mesh = row.mesh
scene = Scene(gl_context, parameters)
# TODO convert all or nothing in BOP.jl? Mesh vs. Pose. Full scene probably not because of cropping
@reset scene.meshes[1].pose = MCMCDepth.to_pose(row.cam_t_m2c, row.cam_R_m2c)

rendered = draw(gl_context, scene)
plot_depth_img(rendered)

# TODO Goal: render the gt pose on top of the recorded (color) image.
