# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using FileIO
using MCMCDepth
using PoseErrors
using SciGL
import CairoMakie as MK

diss_defaults()
begin
    df = gt_targets(joinpath("data", "bop", "tless", "test_primesense"), 18)
    row = df[204, :]
    color_img = load(row.color_path)'

    parameters = Parameters()
    @reset parameters.width = size(color_img)[1]
    @reset parameters.height = size(color_img)[2]
    gl_context = render_context(parameters)

    camera = row.cv_camera
    mesh = upload_mesh(gl_context, load_mesh(row))
    @reset mesh.pose = to_pose(row.gt_t, row.gt_R)
    scene = Scene(row.cv_camera, [mesh])
    aspect_ratio = parameters.width / parameters.height
    fig_width = 0.5 * DISS_WIDTH
    fig = MK.Figure(resolution=(fig_width, fig_width / aspect_ratio))
    plot_scene_ontop!(fig, gl_context, scene, color_img; aspect=aspect_ratio)
    # display(fig)
    MK.save(joinpath("plots", "full_scene.pdf"), fig)
    display(fig)
    destroy_context(gl_context)
end