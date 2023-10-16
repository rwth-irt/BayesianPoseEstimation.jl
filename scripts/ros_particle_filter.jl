# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# TODO test other experiments before merge
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

using ProgressLogging
using TerminalLoggers
using Logging: global_logger
global_logger(TerminalLogger(right_justify=120))

result_dir = datadir("exp_raw", "pf")
bag_name = ["p2_li_0", "p2_li_25_50"]
sampler = [:coordinate_pf, :bootstrap_pf]
posterior = [:simple_posterior, :smooth_posterior]
configs = dict_list(@dict bag_name sampler posterior)

function pf_inference(config)
    # Extract config and load dataset to memory so disk is no bottleneck
    @unpack bag_name, sampler, posterior = config

    # ROS bag
    rosbag_dir = datadir("rosbags", bag_name)
    rosbag = load(joinpath(rosbag_dir, "original.bag"))
    camera = rosbag["/camera/depth/camera_info"] |> first |> CvCamera
    depth_imgs = @. rosbag["/camera/depth/image_rect_raw"] |> ros_depth_img

    # Pose for initialization
    csv_file = joinpath(rosbag_dir, "tf_camera_depth_optical_frame.tracked_object.tum")
    csv_row = CSV.File(csv_file, delim=" ", header=[:timestamp, :tx, :ty, :tz, :qx, :qy, :qz, :qw]) |> first
    t = [csv_row.tx, csv_row.ty, csv_row.tz]
    R = Quaternion(csv_row.qw, csv_row.qx, csv_row.qy, csv_row.qz)

    # File of 3D model for tracking
    mesh_file = joinpath(rosbag_dir, "track.obj")

    # Context
    parameters = Parameters()
    # Coordinate PF evaluates the likelihood twice
    if sampler == :bootstrap_pf
        @reset parameters.n_particles = 1000
    elseif sampler == :coordinate_pf
        @reset parameters.n_particles = 400
    end
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

    # NOTE regularization only makes a difference for association models... Only better for low pixel_σ
    # avoid timing pre-compilation
    eval(sampler)(cpu_rng, dev_rng, eval(posterior), parameters, experiment, diameter, depth_imgs[1:2])
    elaps = @elapsed begin
        states, final_state = eval(sampler)(cpu_rng, dev_rng, eval(posterior), parameters, experiment, diameter, depth_imgs)
    end
    fps = length(depth_imgs) / elaps

    destroy_context(gl_context)
    # For DrWatson
    @strdict states fps parameters
end

# RUN it
@progress "Particle Filters" for config in configs
    @produce_or_load(pf_inference, config, result_dir; filename=c -> savename(c; connector=","))
end

# Convert results to TUM for processing in evo
function parse_config(path)
    config = my_parse_savename(path)
    @unpack bag_name, sampler, posterior = config
    bag_name, sampler, posterior
end
to_secs(time::ROSTime) = Float64(time.secs) + time.nsecs / 1e9
ros_time_secs(msg::MessageData) = to_secs(msg.data.header.time)

raw_df = collect_results(datadir("exp_raw", "pf"))
transform!(raw_df, :path => ByRow(parse_config) => [:bag_name, :sampler, :posterior])

exp_pro = datadir("exp_pro", "pf")
mkpath(exp_pro)

for row in eachrow(raw_df)
    rosbag_dir = datadir("rosbags", row.bag_name)
    rosbag = load(joinpath(rosbag_dir, "original.bag"))

    timestamp = rosbag["/camera/depth/camera_info"] .|> ros_time_secs
    duration = last(timestamp) - first(timestamp)
    bag_fps = length(timestamp) / duration

    x, y, z, qx, qy, qz, qw = [similar(timestamp) for _ in 1:7]
    for (idx, state) in enumerate(row.states)
        _, best_idx = findmax(loglikelihood(state.sample))
        x[idx], y[idx], z[idx] = state.sample.variables.t[:, best_idx]
        qw[idx] = real(state.sample.variables.r[best_idx])
        qx[idx], qy[idx], qz[idx] = imag_part(state.sample.variables.r[best_idx])
    end
    df = DataFrame(timestamp=timestamp, x=x, y=y, z=z, q_x=qx, q_y=qy, q_z=qz, q_w=qw)
    tum_file, _ = row.path |> basename |> splitext
    tum_file *= ".tum"
    CSV.write(joinpath(exp_pro, tum_file), df; delim=" ", writeheader=false)
end

# Plot ESS
begin
    # NOTE looks like smooth_posterior degrades ESS / really focuses on one
    # NOTE coordinate PF has way less sample degeneration
    states = row.states
    MK.lines(1:length(states), exp.(getproperty.(states, :ess)))
end

# Poses ontop of depth image
begin
    rosbag_dir = datadir("rosbags", row.bag_name)
    rosbag = load(joinpath(rosbag_dir, "original.bag"))

    depth_imgs = @. rosbag["/camera/depth/image_rect_raw"] |> ros_depth_img
    parameters = row.parameters
    gl_context = render_context(parameters)
    mesh = upload_mesh(gl_context, joinpath(rosbag_dir, "track.obj"))
    camera = rosbag["/camera/depth/camera_info"] |> first |> CvCamera
    scene = Scene(camera, [mesh])
    experiment = Experiment(gl_context, scene, 0.5, fill(0, 3), one(Quaternion), first(depth_imgs))

    diss_defaults()
    idx = 600
    img = depth_resize(depth_imgs[idx], parameters.width, parameters.height)
    experiment = Experiment(experiment, img)
    depth_img = copy(img)
    depth_min = minimum(depth_img)
    depth_img[depth_img.>1] .= 0
    depth_img = depth_img / maximum(depth_img)
    fig = plot_best_pose(states[idx].sample, experiment, Gray.(depth_img), logprobability)
    display(fig)

    destroy_context(gl_context)
end
