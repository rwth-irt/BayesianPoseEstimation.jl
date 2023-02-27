# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using SciGL

function observation_scene(gl_context, params, occlusion_factor)
    # Camera
    f_x = 120
    f_y = 120
    c_x = 50
    c_y = 50
    camera = CvCamera(params.width, params.height, f_x, f_y, c_x, c_y; near=params.min_depth, far=params.max_depth) |> Camera
    # Meshes
    monkey = load_mesh(gl_context, "meshes/monkey.obj")

    gt_position = params.mean_t + [0.05, -0.05, -0.1]
    gt_orientation = rand(QuaternionUniform())
    rxyz = RotXYZ(QuatRotation(gt_orientation))
    println("GT position & rotation (XYZ): $gt_position, ($(rxyz.theta1), $(rxyz.theta2), $(rxyz.theta3) )")
    @reset monkey.pose = to_pose(gt_position, gt_orientation)

    cube_mesh = load("meshes/cube.obj")
    background_mesh = Scale(3, 3, 1)(cube_mesh)
    background = load_mesh(gl_context, background_mesh)
    @reset background.pose.translation = Translation(0, 0, 3)
    occlusion_mesh = Scale(0.7, 0.7, 0.7)(cube_mesh)
    occlusion = load_mesh(gl_context, occlusion_mesh)
    @reset occlusion.pose.translation = Translation(-0.85 + (0.05 + 0.85) * occlusion_factor, 0, 1.6)

    Scene(camera, [monkey, background, occlusion])
end

function fake_observation(gl_context, params, scene)
    nominal = draw(gl_context, scene)
    # add noise
    association_probability = 0.8f0
    pixel_model = pixel_explicit | (params.min_depth, params.max_depth, params.pixel_θ, params.pixel_σ)
    (; z=rand(device_rng(params), BroadcastedDistribution(pixel_model, (), nominal, association_probability)))
end
