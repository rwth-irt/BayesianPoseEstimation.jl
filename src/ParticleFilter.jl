# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using KernelDistributions
using PoseErrors
using RobotOSData

"""
    coordinate_pf(cpu_rng, dev_rng, posterior_fn, params, experiment, depth_imgs; [collect_vars=(:t, :r)])
Run the particle filter on the `depth_imgs`.
As the model has to be conditioned on new data at every step, a new experiment and posterior model is created at each timestep.

Idea from Block sampling, similar publication: Wüthrich 2015, The Coordinate Particle Filter - a novel Particle Filter for high dimensional systems
"""
function coordinate_pf(cpu_rng::AbstractRNG, dev_rng::AbstractRNG, posterior_fn, params::Parameters, experiment::Experiment, diameter, depth_imgs; collect_vars=(:t, :r))
    state = nothing
    states = Vector{SmcState}()
    for depth_img in depth_imgs
        # Crop image
        if isnothing(state)
            experiment = resize_experiment(experiment, depth_img)
        else
            experiment = resize_experiment(experiment, depth_img)
        end
        prior = pf_prior(params, experiment, cpu_rng)
        posterior = posterior_fn(params, experiment, prior, dev_rng)
        # NOTE component wise sampling is king, running twice allows much lower particle count
        sampler = coordinate_pf_sampler(cpu_rng, params, posterior)
        if isnothing(state)
            _, state = AbstractMCMC.step(cpu_rng, posterior, sampler)
        else
            _, state = AbstractMCMC.step(cpu_rng, posterior, sampler, state)
        end
        push!(states, collect_variables(state, collect_vars))
    end
    states, state
end

"""
    bootstrap_pf(cpu_rng, dev_rng, posterior_fn, params, experiment, depth_imgs; [collect_vars=(:t, :r)])
Run the particle filter on the `depth_imgs`.
As the model has to be conditioned on new data at every step, a new experiment and posterior model is created at each timestep.
"""
function bootstrap_pf(cpu_rng::AbstractRNG, dev_rng::AbstractRNG, posterior_fn, params::Parameters, experiment::Experiment, diameter, depth_imgs; collect_vars=(:t, :r))
    state = nothing
    states = Vector{SmcState}()
    for depth_img in depth_imgs
        experiment = resize_experiment(experiment, depth_img)
        prior = pf_prior(params, experiment, cpu_rng)
        posterior = posterior_fn(params, experiment, prior, dev_rng)
        sampler = bootstrap_pf_sampler(cpu_rng, params, posterior)
        if isnothing(state)
            _, state = AbstractMCMC.step(cpu_rng, posterior, sampler)
        else
            _, state = AbstractMCMC.step(cpu_rng, posterior, sampler, state)
        end
        push!(states, collect_variables(state, collect_vars))
    end
    states, state
end

function crop_experiment(experiment::Experiment, depth_img, t, diameter)
    _, cropped = crop(experiment.scene.camera.object, depth_img, t, diameter)
    width, height = size(experiment.gl_context)
    resized = depth_resize(cropped, width, height)
    Experiment(experiment, resized)
end

function resize_experiment(experiment::Experiment, depth_img)
    width, height = size(experiment.gl_context)
    resized = depth_resize(depth_img, width, height)
    Experiment(experiment, resized)
end

function coordinate_pf_sampler(cpu_rng, params, posterior)
    # tempering does not matter for bootstrap kernel
    temp_schedule = ConstantSchedule()
    t_proposal = Dynamics(:t, cpu_rng, params, posterior)
    r_proposal = Dynamics(:r, cpu_rng, params, posterior)
    t_kernel = BootstrapKernel(t_proposal)
    r_kernel = BootstrapKernel(r_proposal)
    CoordinateSampler(
        SequentialMonteCarlo(t_kernel, temp_schedule, params.n_particles, log(params.relative_ess * params.n_particles)),
        SequentialMonteCarlo(r_kernel, temp_schedule, params.n_particles, log(params.relative_ess * params.n_particles)))
end

function bootstrap_pf_sampler(cpu_rng, params, posterior)
    # tempering does not matter for bootstrap kernel
    temp_schedule = ConstantSchedule()
    # NOTE not component wise
    t_sym = BroadcastedNode(:t, cpu_rng, KernelNormal, 0, params.proposal_σ_t)
    r_sym = BroadcastedNode(:r, cpu_rng, KernelNormal, 0, params.proposal_σ_r)
    # TODO it is possible to use this interface but I really have to bend it to my will... redesign!
    tr_proposal = Proposal(propose_tr_dyn, transition_probability_symmetric, (; t=t_sym, r=r_sym), parents(posterior.prior, :t), (; t=ZeroIdentity(), r=ZeroIdentity()), bijector(posterior))
    tr_kernel = BootstrapKernel(tr_proposal)
    SequentialMonteCarlo(tr_kernel, temp_schedule, params.n_particles, log(params.relative_ess * params.n_particles))
end

"""
    pf_prior(params, experiment, cpu_rng)
Returns a BayesNet for μ(t,r) for an approximately known position and orientation.
Uses the proposal standard deviations.
"""
function pf_prior(params::Parameters, experiment::Experiment, cpu_rng::AbstractRNG)
    t_dot = BroadcastedNode(:t_dot, cpu_rng, KernelNormal, zeros(params.float_type, 3), params.proposal_σ_t)
    r_dot = BroadcastedNode(:r_dot, cpu_rng, KernelNormal, zeros(params.float_type, 3), params.proposal_σ_r)

    t = BroadcastedNode(:t, cpu_rng, KernelNormal, experiment.prior_t, params.proposal_σ_t)
    r = BroadcastedNode(:r, cpu_rng, QuaternionNormal, experiment.prior_r, first(params.proposal_σ_r))

    # include t_dot and r_dot in node but not render function
    μ_fn(t, r, t_dot, r_dot) = render_fn(experiment.gl_context, experiment.scene, t, r)
    DeterministicNode(:μ, μ_fn, (t, r, t_dot, r_dot))
end

"""
    pf_crop_prior(params, experiment, cpu_rng, object_diameter)
Returns a BayesNet for μ(t,r) for an approximately known position and orientation.
Uses the proposal standard deviations.

Crops the images during rendering centered at the current position estimate.
Assumes that the variance of the position estimates is small compared to the object diameter.
# WARN very jittery due to discretization errors maybe another interpolation method would help
"""
function pf_crop_prior(params::Parameters, experiment::Experiment, cpu_rng::AbstractRNG, object_diameter)
    t_dot = BroadcastedNode(:t_dot, cpu_rng, KernelNormal, zeros(params.float_type, 3), params.proposal_σ_t)
    r_dot = BroadcastedNode(:r_dot, cpu_rng, KernelNormal, zeros(params.float_type, 3), params.proposal_σ_r)

    t = BroadcastedNode(:t, cpu_rng, KernelNormal, experiment.prior_t, params.proposal_σ_t)
    r = BroadcastedNode(:r, cpu_rng, QuaternionNormal, experiment.prior_r, first(params.proposal_σ_r))

    # include t_dot and r_dot in node but not render function
    μ_fn(t, r, t_dot, r_dot) = render_crop_fn(experiment.gl_context, experiment.scene, object_diameter, t, r)
    DeterministicNode(:μ, μ_fn, (t, r, t_dot, r_dot))
end

"""
    Dynamics{name}
Proposal which uses state space dynamics for a given variable `name`.
This allows to include custom parameters like the velocity `decay`.
"""
struct Dynamics{name}
    rng
    decay
    σ
    bijectors
    evaluation
end

function Dynamics(name, rng::AbstractRNG, params::Parameters, posterior::PosteriorModel)
    if name == :t
        Dynamics{name}(rng, params.velocity_decay, params.proposal_σ_t, bijector(posterior), parents(posterior.prior, name))
    elseif name == :r
        Dynamics{name}(rng, params.velocity_decay, params.proposal_σ_r, bijector(posterior), parents(posterior.prior, name))
    end
end

transition_probability(dynamics::Dynamics, new_sample, previous_sample) = transition_probability_symmetric(dynamics, new_sample, previous_sample)

function propose(dynamics::Dynamics{:t}, sample, dims...)
    t = sample.variables.t
    t_d = sample.variables.t_dot
    t_dd = rand(dynamics.rng, KernelNormal.(0, dynamics.σ), dims...)
    # Decaying velocity
    @reset sample.variables.t_dot = dynamics.decay * t_d + t_dd
    # Constant acceleration integration
    @reset sample.variables.t = t + t_d + 0.5 * t_dd
    # Evaluate rendering and possibly association
    model_sample, _ = to_model_domain(sample, dynamics.bijectors)
    evaluated = evaluate(dynamics.evaluation, variables(model_sample))
    to_unconstrained_domain(Sample(evaluated), dynamics.bijectors)
end

function propose(dynamics::Dynamics{:r}, sample, dims...)
    r = sample.variables.r
    r_d = sample.variables.r_dot
    r_dd = rand(dynamics.rng, KernelNormal.(0, dynamics.σ), dims...)
    # Decaying velocity
    @reset sample.variables.r_dot = dynamics.decay * r_d + r_dd
    # Constant acceleration integration
    @reset sample.variables.r = r .⊕ (r_d + 0.5 * r_dd)
    # Evaluate rendering
    model_sample, _ = to_model_domain(sample, dynamics.bijectors)
    evaluated = evaluate(dynamics.evaluation, variables(model_sample))
    to_unconstrained_domain(Sample(evaluated), dynamics.bijectors)
end

function propose_tr_dyn(proposal, sample, dims...)
    proposed = rand(proposal.model, dims...)

    t = sample.variables.t
    t_d = sample.variables.t_dot
    t_dd = proposed.t
    # Decaying velocity, hard-coded okay since I only want to reproduce my prior work
    # https://publications.rwth-aachen.de/record/804320
    @reset sample.variables.t_dot = 0.9 * t_d + t_dd
    # Constant acceleration integration
    @reset sample.variables.t = t + t_d + 0.5 * t_dd

    r = sample.variables.r
    r_d = sample.variables.r_dot
    r_dd = proposed.r
    # Decaying velocity
    @reset sample.variables.r_dot = 0.9 * r_d + r_dd
    # Constant acceleration integration
    @reset sample.variables.r = r .⊕ (r_d + 0.5 * r_dd)

    # Evaluate rendering
    model_sample, _ = to_model_domain(sample, proposal.posterior_bijectors)
    evaluated = evaluate(proposal.evaluation, variables(model_sample))
    to_unconstrained_domain(Sample(evaluated), proposal.posterior_bijectors)
end

"""
    CvCamera(camera_info)
Generate a CvCamera from a ROS camera info message.
"""
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

"""
    ros_depth_img(msg)
Extract a depth image from a ROS image message.
"""
function ros_depth_img(msg::MessageData{RobotOSData.CommonMsgs.sensor_msgs.Image})
    width = msg.data.width
    height = msg.data.height
    if msg.data.encoding == "16UC1"
        # millimeters to meters
        img = reinterpret(UInt16, msg.data.data) / 1000.0f0
    elseif msg.data.encoding == "32FC1"
        img = reinterpret(Float32, msg.data.data)
    end
    reshape(img, width, height)
end

"""
    ros_pose(msg)
Extract the translation vector and quaternion as tuple (t, R) from a ROS poses stamped message. 
"""
function ros_pose(msg::MessageData{RobotOSData.CommonMsgs.geometry_msgs.PoseStamped})
    qw = msg.data.pose.orientation.w
    qx = msg.data.pose.orientation.x
    qy = msg.data.pose.orientation.y
    qz = msg.data.pose.orientation.z
    q = normalize(Quaternion(qw, qx, qy, qz))
    tx = msg.data.pose.position.x
    ty = msg.data.pose.position.y
    tz = msg.data.pose.position.z
    [tx, ty, tz], q
end
