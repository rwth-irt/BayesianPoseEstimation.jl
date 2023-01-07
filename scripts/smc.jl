# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using AbstractMCMC: step
using Accessors
using CUDA
using MCMCDepth
using Random
using Plots
using ProgressLogging

gr()
MCMCDepth.diss_defaults()

parameters = Parameters()
# NOTE takes 3min instead of 3sec
# parameters = @set parameters.device = :CPU
gl_context = render_context(parameters)

function fake_observation(parameters::Parameters, gl_context::OffscreenContext, occlusion::Real)
    # Nominal scene
    obs_params = @set parameters.mesh_files = ["meshes/monkey.obj", "meshes/cube.obj", "meshes/cube.obj"]
    obs_scene = Scene(obs_params, gl_context)
    # Background
    obs_scene = @set obs_scene.meshes[2].pose.translation = Translation(0, 0, 3)
    obs_scene = @set obs_scene.meshes[2].scale = Scale(3, 3, 1)
    # Occlusion
    obs_scene = @set obs_scene.meshes[3].pose.translation = Translation(-0.85 + (0.05 + 0.85) * occlusion, 0, 1.6)
    obs_scene = @set obs_scene.meshes[3].scale = Scale(0.7, 0.7, 0.7)
    obs_μ = render(gl_context, obs_scene, parameters.object_id, to_pose(parameters.mean_t + [0.05, -0.05, -0.1], [0, 0, 0]))
    # add noise
    pixel_model = pixel_explicit | (parameters.min_depth, parameters.max_depth, parameters.pixel_θ, parameters.pixel_σ)
    (; z=rand(device_rng(parameters), BroadcastedDistribution(pixel_model, (), obs_μ, 0.8f0)))
end

observation = fake_observation(parameters, gl_context, 0.4)

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
    r = BroadcastedNode(:r, rng, QuaternionUniform, parameters.precision)

    μ_fn = render_fn | (render_context, Scene(parameters, render_context), parameters.object_id)
    μ = DeterministicNode(:μ, μ_fn, (; t=t, r=r))

    dist_is = valid_pixel_normal | parameters.association_σ
    dist_not = smooth_valid_tail | (parameters.min_depth, parameters.max_depth, parameters.pixel_θ, parameters.association_σ)
    association_fn = pixel_association | (dist_is, dist_not, parameters.prior_o)
    o = DeterministicNode(:o, (expectation) -> association_fn.(expectation, observation.z), (; μ=μ))
    # NOTE almost no performance gain over DeterministicNode
    # o = BroadcastedNode(:o, dev_rng, KernelDirac, parameters.prior_o)

    # NOTE valid_pixel diverges without normalization
    pixel_model = smooth_valid_mixture | (parameters.min_depth, parameters.max_depth, parameters.pixel_θ, parameters.pixel_σ)
    z = BroadcastedNode(:z, dev_rng, pixel_model, (; μ=μ, o=o))
    z_norm = ModifierNode(z, dev_rng, ImageLikelihoodNormalizer | parameters.normalization_constant)

    posterior = PosteriorModel(z_norm, observation)

    # Assemble samplers
    # temp_schedule = ExponentialSchedule(n_steps, 0.9999)
    # NOTE LinearSchedule seems reasonable
    temp_schedule = LinearSchedule(n_steps)

    ind_proposal = independent_proposal((; t=t, r=r), z)
    ind_mh_kernel = MhKernel(rng, ind_proposal)
    ind_smc_mh = SequentialMonteCarlo(ind_mh_kernel, temp_schedule, n_particles, log(parameters.relative_ess * n_particles))

    t_sym = BroadcastedNode(:t, rng, KernelNormal, 0, parameters.proposal_σ_t)
    r_sym = BroadcastedNode(:r, rng, QuaternionPerturbation, parameters.proposal_σ_r_quat)
    sym_proposal = symmetric_proposal((; t=t_sym, r=r_sym), z)

    sym_fp_kernel = ForwardProposalKernel(sym_proposal)
    sym_smc_fp = SequentialMonteCarlo(sym_fp_kernel, temp_schedule, n_particles, log(parameters.relative_ess * n_particles))

    sym_mh_kernel = MhKernel(rng, sym_proposal)
    sym_smc_mh = SequentialMonteCarlo(sym_mh_kernel, temp_schedule, n_particles, log(parameters.relative_ess * n_particles))

    sym_boot_kernel = BootstrapKernel(sym_proposal)
    sym_smc_boot = SequentialMonteCarlo(sym_boot_kernel, temp_schedule, n_particles, log(parameters.relative_ess * n_particles))

    # NOTE ind_smc only makes sense when using a MCMCKernel, otherwise I throw away all the information
    composed_sampler = ComposedSampler(Weights([0.1, 1.0]), ind_smc_mh, sym_smc_mh)

    sampler = composed_sampler
    # sampler = sym_smc_mh
    # sampler = sym_smc_fp
    # NOTE tends to diverge with to few samples, since there is no prior pulling it back to sensible values. But it can also converge to very precise values since there is no prior holding it back.
    # sampler = sym_smc_boot

    sample, state = step(rng, posterior, sampler)
    @progress for n in 1:n_steps
        sample, state = step(rng, posterior, sampler, state)
    end
    sample, state
end

# NOTE SMC: tempering is essential. More steps (MCMC) allows higher normalization_constant than more particles (FP, Bootstrap), 15-30 seems to be a good range
parameters = @set parameters.normalization_constant = 20;
parameters = @set parameters.proposal_σ_r_quat = 0.1;
parameters = @set parameters.proposal_σ_t = [0.01, 0.01, 0.01];
parameters = @set parameters.seed = rand(RandomDevice(), UInt32);
# Normalization and tempering leads to less resampling
parameters = @set parameters.relative_ess = 0.8;
# NOTE resampling dominated like FP & Bootstrap kernels typically perform better with more samples (1_000,100) while MCMC kernels tend to perform better with more steps (2_000,50)
final_sample, final_state = run_inference(parameters, gl_context, observation, 1_000, 100);

println("Final log-evidence: $(final_state.log_evidence)")
plot_pose_density(final_sample; trim=false)

anim = @animate for i ∈ 0:2:360
    scatter_position(final_sample, 100, label="particle number", camera=(i, 25), projection_type=:perspective, legend_position=:topright)
end;
gif(anim, "anim_fps15.gif", fps=20)
# TODO diagnostics: Accepted steps, resampling steps
