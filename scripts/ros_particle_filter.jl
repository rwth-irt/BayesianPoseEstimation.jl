# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
Runs the particle filters on the at - Automatisierungstechnik test set.
Running the evo_traj shell command gets stuck when used in run.sh.
Since this script does not take too long, it should not matter. 
"""

using DrWatson
@quickactivate("MCMCDepth")

using Accessors
using CSV
using DataFrames
using FileIO
using LinearAlgebra
using MCMCDepth
using PoseErrors
using Printf
using Quaternions
using Random
using RobotOSData
using SciGL
using StatsBase
import CairoMakie as MK

using ProgressLogging
using TerminalLoggers
using Logging: global_logger
global_logger(TerminalLogger(right_justify=120))

result_dir = datadir("exp_raw", "pf")
# TODO p2_li_0 only contains half the data?
bag_name = "p2_li_50_95"  # ["p2_li_25_50", "p2_li_50_95"]
# NOTE coordinate a bit more stable (association p2_li_25_50) otherwise no big difference?
# WARN do not crop - shaky due to discretization error
sampler = [:coordinate_pf, :bootstrap_pf]
posterior = [:simple_posterior, :smooth_posterior]
prior_o = [0.49, 0.51]
configs = dict_list(@dict bag_name sampler posterior prior_o)

parameters = Parameters()
@reset parameters.width = 80
@reset parameters.height = 60
@reset parameters.depth = 1_500
gl_context = render_context(parameters)
@reset parameters.relative_ess = 0.5
# NOTE low value crucial for best performance
@reset parameters.pixel_σ = 0.001
@reset parameters.association_σ = parameters.pixel_σ
@reset parameters.min_depth = 0.15
@reset parameters.max_depth = 10
@reset parameters.proposal_σ_t = fill(1e-3, 3)
@reset parameters.proposal_σ_r = fill(1e-3, 3)
@reset parameters.velocity_decay = 0.9

function pf_inference(config, gl_context, parameters)
    # Extract config and load dataset to memory so disk is no bottleneck
    @unpack bag_name, sampler, posterior, prior_o = config
    prior_o = parameters.float_type(prior_o)

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

    # Coordinate PF evaluates the likelihood twice
    # Targets 90Hz of Intel Realsense cameras
    if sampler == :bootstrap_pf
        @reset parameters.n_particles = 1250
    elseif sampler == :coordinate_pf
        @reset parameters.n_particles = 600
    end
    cpu_rng = Random.default_rng(parameters)
    dev_rng = device_rng(parameters)

    # File of 3D model for tracking
    mesh_file = joinpath(rosbag_dir, "track.obj")
    mesh = load(mesh_file)
    diameter = model_diameter(mesh)
    scene_mesh = upload_mesh(gl_context, mesh)
    scene = Scene(camera, [scene_mesh])
    experiment = Experiment(gl_context, scene, prior_o, t, R, first(depth_imgs))

    # NOTE regularization only makes a difference for association models
    # avoid timing pre-compilation
    eval(sampler)(cpu_rng, dev_rng, eval(posterior), parameters, experiment, diameter, depth_imgs[1:2])
    elaps = @elapsed begin
        states, final_state = eval(sampler)(cpu_rng, dev_rng, eval(posterior), parameters, experiment, diameter, depth_imgs)
    end
    fps = length(depth_imgs) / elaps

    # For DrWatson
    @strdict states fps parameters
end

# RUN it
pf_closure(config) = pf_inference(config, gl_context, parameters)
@progress "Particle Filters" for config in configs
    @produce_or_load(pf_closure, config, result_dir; filename=c -> savename(c; connector=","))
end
destroy_context(gl_context)

# Convert results to TUM for processing in evo
function parse_config(path)
    config = my_parse_savename(path)
    @unpack bag_name, sampler, posterior, prior_o = config
    bag_name, sampler, posterior, prior_o
end
to_secs(time::ROSTime) = Float64(time.secs) + time.nsecs / 1e9
ros_time_secs(msg::MessageData) = to_secs(msg.data.header.time)
raw_df = collect_results(datadir("exp_raw", "pf"))
transform!(raw_df, :path => ByRow(parse_config) => [:bag_name, :sampler, :posterior, :prior_o])

# Setup python environment for evo
run(`bash -c """cd scripts/rosbag \
&& python3 -m venv venv \
&& source venv/bin/activate \
&& pip install -r requirements.txt"""`)

exp_pro = datadir("exp_pro", "pf")
mkpath(exp_pro)

function pf_title(bag_name, sampler, posterior, prior_o, fps)
    occ_string = sampler_str = model_str = ""
    if contains(bag_name, "25_50")
        occ_string = "25-50% occlusion, "
    elseif contains(bag_name, "50_95")
        occ_string = "50-95% occlusion, "
    end
    if sampler == "bootstrap_pf"
        sampler_str = "joint, "
    elseif sampler == "coordinate_pf"
        sampler_str = "block-wise, "
    end
    if posterior == "simple_posterior"
        model_str = "simple, "
    elseif posterior == "smooth_posterior"
        model_str = "complex, "
    end
    occ_string * sampler_str * model_str * "p(cₒ)=$(prior_o), " * "$(round(Int, fps))Hz"
end

for row in eachrow(raw_df)
    rosbag_dir = datadir("rosbags", row.bag_name)
    bag_file = joinpath(rosbag_dir, "original.bag")
    rosbag = load(bag_file)

    timestamp = rosbag["/camera/depth/camera_info"] .|> ros_time_secs
    duration = last(timestamp) - first(timestamp)
    bag_fps = length(timestamp) / duration

    x, y, z, qx, qy, qz, qw = [similar(timestamp) for _ in 1:7]
    for (idx, state) in enumerate(row.states)
        t_mean = mean(state.sample.variables.t, weights(exp.(state.log_weights)); dims=2)
        q_mean = mean(state.sample.variables.r, weights(exp.(state.log_weights)))
        x[idx], y[idx], z[idx] = t_mean
        qw[idx] = real(q_mean)
        qx[idx], qy[idx], qz[idx] = imag_part(q_mean)
    end
    df = DataFrame(timestamp=timestamp, x=x, y=y, z=z, q_x=qx, q_y=qy, q_z=qz, q_w=qw)
    tum_file, _ = row.path |> basename |> splitext
    tum_file *= ".tum"
    tum_file = joinpath(exp_pro, tum_file)
    println(tum_file)
    CSV.write(tum_file, df; delim=" ", writeheader=false)

    # combine results
    run(`bash -c "cd scripts/rosbag && source venv/bin/activate && python tf_bag.py $bag_file $tum_file"`)

    # Plot the results
    julia_tum = "scripts/rosbag/tf_world.julia_pf.tum"
    baseline_tum = "scripts/rosbag/tf_world.tracked_object.tum"
    isfile(julia_tum) ? rm(julia_tum) : nothing
    isfile(baseline_tum) ? rm(baseline_tum) : nothing

    result_bag = first(splitext(tum_file)) * ".bag"
    run(`bash -c "cd scripts/rosbag && source venv/bin/activate \
        && source /opt/ros/noetic/setup.bash \
        && evo_traj bag  $result_bag /tf:world.julia_pf \
        --save_as_tum --config evo_config.json"`)

    stamp_julia, t_julia, R_julia = load_tum("scripts/rosbag/tf_world.julia_pf.tum")
    stamp_julia = stamp_julia .- first(stamp_julia)
    t_julia = t_julia .- (first(t_julia),)
    t_err_julia = norm.(t_julia)
    R_err_julia = quat_dist.(first(R_julia), R_julia)
    stamp_robot, t_robot, R_robot = load_tum(joinpath("data", "rosbags", row.bag_name, "pose_robot_pf.tum"))
    stamp_robot = stamp_robot .- first(stamp_robot)
    t_robot = t_robot .- (first(t_robot),)
    t_err_robot = norm.(t_robot)
    R_err_robot = quat_dist.(first(R_robot), R_robot)
    stamp_only, t_only, R_only = load_tum(joinpath("data", "rosbags", row.bag_name, "pose_only_pf.tum"))
    stamp_only, t_only, R_only = stamp_only[5:end], t_only[5:end], R_only[5:end]
    stamp_only = stamp_only .- first(stamp_only)
    t_only = t_only .- (first(t_only),)
    t_err_only = norm.(t_only)
    R_err_only = quat_dist.(first(R_only), R_only)

    # Plot em
    diss_defaults()
    fig = MK.Figure(resolution=(DISS_WIDTH, 0.55 * DISS_WIDTH))

    # orientation
    ax = MK.Axis(fig[2, 1]; xlabel="time / s", ylabel="error / °")
    MK.lines!(ax, stamp_robot, rad2deg.(R_err_robot); label="previous robot")
    MK.lines!(ax, stamp_only, rad2deg.(R_err_only); label="previous only")
    MK.lines!(ax, stamp_julia, rad2deg.(R_err_julia); label="smc pf")

    # ESS plots
    # NOTE looks like smooth_posterior degrades ESS / really focuses on one
    # NOTE coordinate PF has way less sample degeneration
    states = row.states
    ax = MK.Axis(fig[2, 2]; xlabel="iteration", ylabel="relative ESS", limits=(nothing, (0, 1)))
    MK.lines!(ax, 1:length(states), exp.(getproperty.(states, :log_relative_ess)); label="smc pf")
    MK.axislegend(ax, position=:lt)

    # position
    ax = MK.Axis(fig[1, :]; ylabel="error / mm", title=pf_title(row.bag_name, row.sampler, row.posterior, row.prior_o, row.fps))
    MK.lines!(ax, stamp_robot, t_err_robot * 1e3; label="at robot")
    MK.lines!(ax, stamp_only, t_err_only * 1e3; label="at only")
    MK.lines!(ax, stamp_julia, t_err_julia * 1e3; label="smc pf")
    MK.axislegend(ax, position=:lt)


    # Final adjustment, display, save
    MK.colsize!(fig.layout, 2, MK.Auto(0.5))
    # display(fig)
    mkpath(joinpath("plots", "pf"))
    save(joinpath("plots", "pf", "$(row.bag_name)_$(row.sampler)_$(row.posterior)_$(row.prior_o).pdf"), fig)
    # Remove files
    isfile(julia_tum) ? rm(julia_tum) : nothing
    isfile(baseline_tum) ? rm(baseline_tum) : nothing
end
