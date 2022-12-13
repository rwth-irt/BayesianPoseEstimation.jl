# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using AbstractMCMC
using Accessors
using CUDA
using Distributions
using MCMCDepth
using Random
using Plots
using Plots.PlotMeasures

pyplot()
MCMCDepth.diss_defaults(; fontfamily="Carlito", fontsize=11, markersize=2.5, size=(160, 90))

parameters = Parameters()
render_context = RenderContext(parameters.width, parameters.height, parameters.depth, device_array_type(parameters))

function fake_observation(parameters::Parameters, render_context::RenderContext, occlusion::Real)
    # Nominal scene
    obs_params = @set parameters.mesh_files = ["meshes/monkey.obj", "meshes/cube.obj", "meshes/cube.obj"]
    obs_scene = Scene(obs_params, render_context)
    # Background
    obs_scene = @set obs_scene.meshes[2].pose.translation = Translation(0, 0, 3)
    obs_scene = @set obs_scene.meshes[2].scale = Scale(3, 3, 1)
    # Occlusion
    obs_scene = @set obs_scene.meshes[3].pose.translation = Translation(-0.85 + (0.05 + 0.85) * occlusion, 0, 1.6)
    obs_scene = @set obs_scene.meshes[3].scale = Scale(0.7, 0.7, 0.7)
    obs_μ = render(render_context, obs_scene, parameters.object_id, to_pose(parameters.mean_t + [0.05, -0.05, -0.1], [0, 0, 0]))
    # add noise
    pixel_model = pixel_explicit | (parameters.min_depth, parameters.max_depth, parameters.pixel_θ, parameters.pixel_σ)
    (; z=rand(device_rng(parameters), BroadcastedDistribution(pixel_model, (), obs_μ, 0.8f0)))
end

observation = fake_observation(parameters, render_context, 0.4)

function plot_pose_chain(model_chain, step=200)
    plt_t_chain = plot_variable(model_chain, :t, step; label=["x" "y" "z"], xlabel="Iteration [÷ $(step)]", ylabel="Position [m]", legend=false)
    plt_t_dens = density_variable(model_chain, :t; label=["x" "y" "z"], xlabel="Position [m]", ylabel="Wahrscheinlichkeit", legend=false, left_margin=5mm)

    plt_r_chain = plot_variable(model_chain, :r, step; label=["x" "y" "z"], xlabel="Iteration [÷ $(step)]", ylabel="Orientierung [rad]", legend=false, top_margin=5mm)
    plt_r_dens = density_variable(model_chain, :r; label=["x" "y" "z"], xlabel="Orientierung [rad]", ylabel="Wahrscheinlichkeit", legend=false)

    plot(
        plt_t_chain, plt_r_chain,
        plt_t_dens, plt_r_dens,
        layout=(2, 2)
    )
end

function run_inference(parameters::Parameters, render_context, observation, n_steps=1_000, n_particles=500; kwargs...)
    # Device
    if parameters.device === :CUDA
        CUDA.allowscalar(false)
    end
    # RNGs
    rng = cpu_rng(parameters)
    dev_rng = device_rng(parameters)

    # Model specification
    t = BroadcastedNode(:t, rng, KernelNormal, parameters.mean_t, parameters.σ_t)
    r = BroadcastedNode(:r, rng, QuaternionDistribution, parameters.precision)

    μ_fn = render_fn | (render_context, Scene(parameters, render_context), parameters.object_id)
    μ = DeterministicNode(:μ, μ_fn, (; t=t, r=r))

    dist_is = valid_pixel_normal | parameters.association_σ
    dist_not = valid_pixel_tail | (parameters.min_depth, parameters.max_depth, parameters.pixel_θ)
    association_fn = pixel_association | (dist_is, dist_not, parameters.prior_o)
    o = DeterministicNode(:o, (expectation) -> association_fn.(expectation, observation.z), (; μ=μ))

    # NOTE valid_pixel diverges without normalization
    pixel_model = valid_pixel_mixture | (parameters.min_depth, parameters.max_depth, parameters.pixel_θ, parameters.pixel_σ)
    z = BroadcastedNode(:z, dev_rng, pixel_model, (; μ=μ, o=o))
    z_norm = ModifierNode(z, dev_rng, ImageLikelihoodNormalizer | parameters.normalization_constant)

    posterior = PosteriorModel(z_norm, observation)

    # Assemble samplers
    # TODO thinning
    # temp_schedule = ConstantSchedule()
    temp_schedule = ExponentialSchedule(n_steps, 0.9999)
    # temp_schedule = LinearSchedule(n_steps)

    t_sym = symmetric_proposal(BroadcastedNode(:t, rng, KernelNormal, 0, parameters.proposal_σ_t), z)
    t_smc_fp = ForwardProposalKernel(t_sym)
    t_smc_mh = MhKernel(t_sym, rng)
    # TODO parameter for ESS
    t_smc = SequentialMonteCarlo(t_smc_fp, temp_schedule, n_particles, log(0.5 * n_particles))
    # t_smc = SequentialMonteCarlo(t_smc_mh, temp_schedule, n_particles, log(0.5 * n_particles))

    r_sym = quaternion_symmetric(BroadcastedNode(:r, rng, QuaternionPerturbation, parameters.proposal_σ_r_quat), z)
    r_smc_fp = ForwardProposalKernel(r_sym)
    r_smc_mh = MhKernel(r_sym, rng)
    r_smc = SequentialMonteCarlo(r_smc_fp, temp_schedule, n_particles, log(0.5 * n_particles))
    # r_smc = SequentialMonteCarlo(r_smc_mh, temp_schedule, n_particles, log(0.5 * n_particles))

    # ComposedSampler
    # TODO resampling step destroys the distribution of the other variable
    composed_sampler = ComposedSampler(Weights([1.0, 1.0]), t_smc, r_smc)
    # composed_sampler = ComposedSampler(Weights([0.1, 1.0, 0.1, 1.0]), t_ind_mh, r_ind_mh, t_sym_mh, r_sym_mh)

    # WARN random acceptance needs to be calculated on CPU, thus CPU rng
    sample(rng, posterior, composed_sampler, n_steps; discard_initial=0_000, thinning=1, kwargs...)
end

# plot_depth_img(Array(obs.z))
# NOTE SMC: tempering is essential? Use higher normalization_constant since it will be tempered
parameters = @set parameters.normalization_constant = 20
parameters = @set parameters.proposal_σ_r_quat = 0.2
parameters = @set parameters.proposal_σ_t = [0.02, 0.02, 0.02]
parameters = @set parameters.seed = rand(RandomDevice(), UInt32)
# TODO out of memory, Do not use AbstractMCMC?
chain = run_inference(parameters, render_context, observation, 200, 100; thinning=1);
plot([s.log_evidence for s in chain])
final = last(chain).sample;
density(transpose(final.variables.t); fill=true, fillalpha=0.4, trim=true)
M = map(final.variables.r) do q
    r_q = QuatRotation(q)
    r_xyz = RotXYZ(r_q)
    [r_xyz.theta1, r_xyz.theta2, r_xyz.theta3]
end;
M = hcat(M...);
density(M'; fill=true, fillalpha=0.4, trim=true)
