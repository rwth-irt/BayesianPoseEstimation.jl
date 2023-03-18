# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using FileIO
using ImageTransformations
using MCMCDepth
using Plots
using SciGL

# s_df = scene_dataframe("tless", "test_primesense", 1)
# s_df = scene_dataframe("itodd", "val", 1)
s_df = scene_dataframe("lm", "test", 2)
row = s_df[100, :]
width, height = row.img_size

parameters = Parameters()
# Context
@reset parameters.width = width
@reset parameters.height = height
@reset parameters.depth = 10

gl_context = render_context(parameters)

# Scene
cv_camera = row.camera
mesh = upload_mesh(gl_context, row.mesh)
@reset mesh.pose = to_pose(row.cam_t_m2c, row.cam_R_m2c)
scene = Scene(Camera(cv_camera), [mesh])

# Draw result for visual validation
MCMCDepth.diss_defaults()
gr()
rendered_img = draw(gl_context, scene)
color_img = load(row.color_path)
plot_depth_ontop(color_img, rendered_img)

# Crop
@reset parameters.width = 400
@reset parameters.height = 400
@reset parameters.depth = 1
bounding_box = crop_boundingbox(cv_camera, row.cam_t_m2c, row.diameter)
crop_img = crop_image(color_img, bounding_box..., parameters)

gl_context = render_context(parameters)
mesh = upload_mesh(gl_context, row.mesh)
@reset mesh.pose = to_pose(row.cam_t_m2c, row.cam_R_m2c)
crop_camera = crop(cv_camera, bounding_box...)
crop_scene = Scene(crop_camera, [mesh])

crop_render = draw(gl_context, crop_scene)
plot_depth_ontop(crop_img, crop_render, alpha=0.8)

mask_img = Bool.(load(row.mask_path))
crop_mask = crop_image(mask_img, bounding_box..., parameters)

# TODO julia images vs. OpenGL (width, height)
depth_img = load_depth_image(row)
crop_depth = crop_image(depth_img, bounding_box..., parameters)
plot_depth_img((crop_depth .* crop_mask)')

# TODO load camera noise depending on dataset name? Probabilistic Robotics: Larger Noise than expected? Tune parameter?
