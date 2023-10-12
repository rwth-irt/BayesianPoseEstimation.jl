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

WIDTH = HEIGHT = 50

# TODO document in README how BOP and ros datasets must be stored in data/
# make sure to decompress the data or it will be painfully slow
img_bag = load("data/p2_li_25_50/p2_li_25_50.bag")

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
depth_img[depth_img.>2.5] .= 0
plot_depth_img(depth_img)

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
pose_bag = load("data/p2_li_25_50/p2_li_25_50_poses.bag")
t, R = pose_bag["/tf/camera_depth_optical_frame.filtered_object"] |> last |> ros_pose
pose = to_pose(t, R)

# Context
parameters = Parameters()
@reset parameters.width = 100
@reset parameters.height = 100
@reset parameters.n_particles = 100;

# TODO Evaluate different numbers of particles
cpu_rng = Random.default_rng(parameters)
dev_rng = device_rng(parameters)

# TODO do not hardcode depth
gl_context = render_context(parameters)
mesh = upload_mesh(gl_context, "data/p2_li_25_50/track.obj")
@reset mesh.pose = pose
scene = Scene(camera, [mesh])

# Preview decoded image and pose
# rendered_img = draw(gl_context, scene)
# plot_depth_img(rendered_img) |> display
# plot_depth_img(depth_img) |> display

# Pre-load data for particle filtering so disk reading is not the bottleneck
# TODO export depth_resize
resize_closure(img) = PoseErrors.depth_resize(img, parameters.width, parameters.height)
depth_imgs = @. img_bag["/camera/depth/image_rect_raw"] |> ros_depth_img |> resize_closure


# Prepare
# TODO use prior from previous image?
prior_o = fill(parameters.float_type(0.5), parameters.width, parameters.height) |> device_array_type(parameters)
# Prior t only used for initialization if using bootstrap kernel
prior_t = t


# Filter loop
state = nothing
elaps = @elapsed for depth_img in depth_imgs
    # TODO crop depth_img
    # Online execution would also require transfer to device
    device_img = depth_img |> device_array_type(parameters)
    # TODO it works but is quite hidden... Should it be default Experiment constructor?
    experiment = preprocessed_experiment(gl_context, scene, prior_o, prior_t, device_img)
    # Model
    prior = point_prior(parameters, experiment, cpu_rng)
    # TODO or association / smooth
    posterior = simple_posterior(parameters, experiment, prior, dev_rng)
    # Bootstrap kernel for particle filter
    sampler = smc_bootstrap(cpu_rng, parameters, posterior)
    # TODO
    if isnothing(state)
        _, state = AbstractMCMC.step(cpu_rng, posterior, sampler)
    else
        _, state = AbstractMCMC.step(cpu_rng, posterior, sampler, state)
    end
end
frame_rate = length(depth_imgs) / elaps

diss_defaults()
fig = plot_pose_density(state.sample)
fig, _, _ = plot_best_pose(state.sample, experiment, color_img, logprobability)
