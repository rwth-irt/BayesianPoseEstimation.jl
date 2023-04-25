# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using FileIO
using SciGL

fake_gt_position = [0, 0, 2.0]
fake_mesh = load("meshes/monkey.obj")

function fake_camera(params)
    f_x = 1.2 * params.width
    f_y = f_x
    c_x = 0.5 * params.width
    c_y = 0.5 * params.height
    CvCamera(params.width, params.height, f_x, f_y, c_x, c_y) |> Camera
end

function fake_observation(gl_context, params, occlusion_factor)
    monkey = upload_mesh(gl_context, fake_mesh)
    gt_orientation = rand(QuaternionUniform())
    rxyz = RotXYZ(QuatRotation(gt_orientation))
    println("GT position & rotation (XYZ): $fake_gt_position, ($(rxyz.theta1), $(rxyz.theta2), $(rxyz.theta3) )")
    @reset monkey.pose = to_pose(fake_gt_position, gt_orientation)

    cube_mesh = load("meshes/cube.obj")
    background_mesh = Scale(3, 3, 1)(cube_mesh)
    background = upload_mesh(gl_context, background_mesh)
    @reset background.pose.translation = Translation(0, 0, 3)
    occlusion_mesh = Scale(0.7, 0.7, 0.7)(cube_mesh)
    occlusion = upload_mesh(gl_context, occlusion_mesh)
    @reset occlusion.pose.translation = Translation(-0.85 + (0.05 + 0.85) * occlusion_factor, 0, 1.6)

    scene = Scene(fake_camera(params), [monkey, background, occlusion])
    nominal = draw(gl_context, scene)
    # add noise
    association_probability = 0.8f0
    pixel_model = pixel_explicit | (params.min_depth, params.max_depth, params.pixel_θ, params.pixel_σ)
    rand(device_rng(params), BroadcastedDistribution(pixel_model, (), nominal, association_probability))
end
