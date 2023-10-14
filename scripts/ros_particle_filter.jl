# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using FileIO
using LinearAlgebra
using MCMCDepth
using PoseErrors
using Quaternions
using Random
using RobotOSData
using SciGL

import CairoMakie as MK

# TODO document in README how BOP and ros datasets must be stored in data/
# TODO doc make sure to decompress the data or it will be painfully slow
# TODO as variable
img_bag = load("data/p2_li/p2_li_25_50.bag")

function SciGL.CvCamera(camera_info=MessageData{RobotOSData.CommonMsgs.sensor_msgs.CameraInfo})
    K = camera_info.data.K
    width = camera_info.data.width
    height = camera_info.data.height
    fx = K[1]
    sk = K[2]
    cx = K[3]
    fy = K[5]
    cy = K[6]
    CvCamera(width, height, fx, fy, cx, cy; s=sk)
end
camera = img_bag["/camera/depth/camera_info"] |> first |> CvCamera |> Camera
# TODO Implement dynamic cropping using center_diameter_boundingbox from PoseErrors

function ros_depth_img(depth_img::MessageData{RobotOSData.CommonMsgs.sensor_msgs.Image})
    width = depth_img.data.width
    height = depth_img.data.height
    if depth_img.data.encoding == "16UC1"
        # millimeters to meters
        img = reinterpret(UInt16, depth_img.data.data) / 1000.0f0
    elseif depth_img.data.encoding == "32FC1"
        img = reinterpret(Float32, depth_img.data.data)
    end
    reshape(img, width, height)
end
depth_img = img_bag["/camera/depth/image_rect_raw"] |> first |> ros_depth_img
# filter outliers for visualization
# depth_img[depth_img.>2.5] .= 0
# plot_depth_img(depth_img)

function ros_pose(pose_msg::MessageData{RobotOSData.CommonMsgs.geometry_msgs.PoseStamped})
    qw = pose_msg.data.pose.orientation.w
    qx = pose_msg.data.pose.orientation.x
    qy = pose_msg.data.pose.orientation.y
    qz = pose_msg.data.pose.orientation.z
    q = normalize(Quaternion(qw, qx, qy, qz))
    tx = pose_msg.data.pose.position.x
    ty = pose_msg.data.pose.position.y
    tz = pose_msg.data.pose.position.z
    [tx, ty, tz], q
end
pose_bag = load("data/p2_li/p2_li_25_50_poses.bag")
t, R = pose_bag["/tf/camera_depth_optical_frame.filtered_object"] |> first |> ros_pose
pose = to_pose(t, R)

# Context
parameters = Parameters()
# TODO lower to 50 after crop
@reset parameters.width = 100
@reset parameters.height = 100
@reset parameters.depth = 300
gl_context = render_context(parameters)
cpu_rng = Random.default_rng(parameters)
dev_rng = device_rng(parameters)

# Pre-load data for particle filtering so disk reading is not the bottleneck
# TODO export depth_resize
resize_closure(img) = PoseErrors.depth_resize(img, parameters.width, parameters.height)
depth_imgs = @. img_bag["/camera/depth/image_rect_raw"] |> ros_depth_img |> resize_closure

# TODO do not hardcode depth
mesh = upload_mesh(gl_context, "data/p2_li/track.obj")
@reset mesh.pose = pose
scene = Scene(camera, [mesh])

# TODO Evaluate different numbers of particles
@reset parameters.n_particles = parameters.depth;
@reset parameters.relative_ess = 0.5;
# NOTE low value crucial for best performance
@reset parameters.pixel_σ = 0.001
@reset parameters.proposal_σ_t = fill(1e-3, 3)
@reset parameters.proposal_σ_r = fill(1e-3, 3)

# Preview decoded image and pose
# rendered_img = draw(gl_context, scene)
# plot_depth_img(rendered_img) |> display
# plot_depth_img(depth_img) |> display

# Prepare
# TODO use prior from previous image?
prior_o = 0.5f0 #fill(parameters.float_type(0.6), parameters.width, parameters.height) |> device_array_type(parameters)

# Filter loop
state = nothing
experiment = Experiment(gl_context, scene, prior_o, t, R, first(depth_imgs))
elaps = @elapsed begin
    # TODO different posterior_fn
    # NOTE regularization only makes a difference for association models... Only better for low pixel_σ
    states, final_state = MCMCDepth.pf_inference(cpu_rng, dev_rng, simple_posterior, parameters, experiment, depth_imgs)
end
frame_rate = length(depth_imgs) / elaps
# Looks much more reasonable for association models
MK.lines(1:length(states), exp.(getproperty.(states, :ess)))

begin
    diss_defaults()
    idx = 100
    experiment = Experiment(experiment, depth_imgs[idx])
    depth_img = copy(depth_imgs[idx])
    depth_min = minimum(depth_img)
    depth_img[depth_img.>1] .= 0
    depth_img = depth_img / maximum(depth_img)
    fig = plot_best_pose(states[idx].sample, experiment, Gray.(depth_img), logprobability)
    display(fig)
end
# fig = plot_pose_density(final_state.sample)
