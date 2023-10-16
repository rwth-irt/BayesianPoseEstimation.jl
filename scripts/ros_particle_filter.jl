# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# include("../src/MCMCDepth.jl")

using Accessors
using CSV
using DataFrames
using DrWatson
using FileIO
using LinearAlgebra
using MCMCDepth
using PoseErrors
using Quaternions
using Random
using RobotOSData
using SciGL

import CairoMakie as MK

# TODO doc make sure to decompress the data or it will be painfully slow
experiment_name = "p2_li_25_50"
result_dir = datadir("exp_raw", "pf", experiment_name)

# Pre-load data for particle filtering so disk reading is not the bottleneck
# ROS bag
rosbag_dir = datadir("rosbags", experiment_name)
rosbag = load(joinpath(rosbag_dir, "original.bag"))
camera = rosbag["/camera/depth/camera_info"] |> first |> CvCamera
depth_imgs = @. rosbag["/camera/depth/image_rect_raw"] |> ros_depth_img;
# Pose for initialization
csv_file = joinpath(rosbag_dir, "tf_camera_depth_optical_frame.tracked_object.tum")
csv_row = CSV.File(csv_file, delim=" ", header=[:timestamp, :tx, :ty, :tz, :qx, :qy, :qz, :qw]) |> first
t = [csv_row.tx, csv_row.ty, csv_row.tz]
R = Quaternion(csv_row.qw, csv_row.qx, csv_row.qy, csv_row.qz)
# File of 3D model for tracking
mesh_file = joinpath(rosbag_dir, "track.obj")

# Context
parameters = Parameters()
@reset parameters.n_particles = 400
@reset parameters.relative_ess = 0.5
# NOTE low value crucial for best performance
prior_o = 0.5f0
@reset parameters.pixel_σ = 0.001
@reset parameters.proposal_σ_t = fill(1e-3, 3)
@reset parameters.proposal_σ_r = fill(1e-3, 3)

@reset parameters.width = 50
@reset parameters.height = 50
@reset parameters.depth = parameters.n_particles
gl_context = render_context(parameters)

cpu_rng = Random.default_rng(parameters)
dev_rng = device_rng(parameters)

mesh = load(mesh_file)
diameter = model_diameter(mesh)
scene_mesh = upload_mesh(gl_context, mesh)
scene = Scene(camera, [scene_mesh])

experiment = Experiment(gl_context, scene, prior_o, t, R, first(depth_imgs))

elaps = @elapsed begin
    # NOTE regularization only makes a difference for association models... Only better for low pixel_σ
    states, final_state = run_coordinate_pf(cpu_rng, dev_rng, smooth_posterior, parameters, experiment, diameter, depth_imgs)
end
println("tracking rate: $(length(depth_imgs) / elaps)Hz")
# Looks much more reasonable for association models
MK.lines(1:length(states), exp.(getproperty.(states, :ess)))

begin
    diss_defaults()
    idx = 800
    img = depth_resize(depth_imgs[idx], parameters.width, parameters.height)
    experiment = Experiment(experiment, img)
    depth_img = copy(img)
    depth_min = minimum(depth_img)
    depth_img[depth_img.>1] .= 0
    depth_img = depth_img / maximum(depth_img)
    fig = plot_best_pose(states[idx].sample, experiment, Gray.(depth_img), logprobability)
    display(fig)
end

# export TUM
to_secs(time::ROSTime) = Float64(time.secs) + time.nsecs / 1e9
ros_time_secs(msg::MessageData) = to_secs(msg.data.header.time)

timestamp = rosbag["/camera/depth/camera_info"] .|> ros_time_secs
duration = last(timestamp) - first(timestamp)
bag_fps = length(timestamp) / duration

x, y, z, qx, qy, qz, qw = [similar(timestamp) for _ in 1:7];
for (idx, state) in enumerate(states)
    _, best_idx = findmax(loglikelihood(state.sample))
    x[idx], y[idx], z[idx] = state.sample.variables.t[:, best_idx]
    qw[idx] = real(state.sample.variables.r[best_idx])
    qx[idx], qy[idx], qz[idx] = imag_part(state.sample.variables.r[best_idx])
end
# TODO save via DrWatson
df_dict = @dict timestamp x y z qx qy qz qw

df = DataFrame(timestamp=timestamp, x=x, y=y, z=z, q_x=qx, q_y=qy, q_z=qz, q_w=qw)
mkpath(result_dir)
CSV.write(joinpath(result_dir, "coordinate_pf.tum"), df; delim=" ", writeheader=false)
