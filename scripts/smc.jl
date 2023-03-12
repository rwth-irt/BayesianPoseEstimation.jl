# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

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
# NOTE takes minutes instead of seconds
# @reset parameters.device = :CPU
gl_context = render_context(parameters)

include("fake_observation.jl")
camera = fake_camera(parameters)
model = upload_mesh(gl_context, fake_mesh)
prior_t = fake_gt_position + [0.05, -0.05, -0.1]
fake_img = fake_observation(gl_context, parameters, 0.4)

experiment = Experiment(Scene(camera, [model]), prior_t, fake_img)

function run_inference(render_context, params::Parameters, experiment::Experiment; kwargs...)
    # Device
    if params.device === :CUDA
        CUDA.allowscalar(false)
    end
    # RNGs
    rng = cpu_rng(params)
    dev_rng = device_rng(params)

    # Model specification
    t = BroadcastedNode(:t, rng, KernelNormal, experiment.prior_t, params.σ_t)
    r = BroadcastedNode(:r, rng, QuaternionUniform, params.float_type)

    μ_fn = render_fn | (render_context, experiment.scene)
    μ = DeterministicNode(:μ, μ_fn, (; t=t, r=r))

    dist_is = valid_pixel_normal | params.association_σ
    dist_not = smooth_valid_tail | (params.min_depth, params.max_depth, params.pixel_θ, params.association_σ)
    association_fn = pixel_association | (dist_is, dist_not, params.prior_o)
    o = DeterministicNode(:o, (expectation) -> association_fn.(expectation, experiment.depth_image), (; μ=μ))
    # NOTE almost no performance gain over DeterministicNode
    # o = BroadcastedNode(:o, dev_rng, KernelDirac, parameters.prior_o)

    # NOTE valid_pixel diverges without normalization
    pixel_model = smooth_valid_mixture | (params.min_depth, params.max_depth, params.pixel_θ, params.pixel_σ)
    z = BroadcastedNode(:z, dev_rng, pixel_model, (; μ=μ, o=o))
    z_norm = ModifierNode(z, dev_rng, ImageLikelihoodNormalizer | params.normalization_constant)

    posterior = PosteriorModel(z_norm, (; z=experiment.depth_image))

    # Assemble samplers
    # temp_schedule = ExponentialSchedule(params.n_steps, 0.9999)
    # NOTE LinearSchedule seems reasonable
    temp_schedule = LinearSchedule(params.n_steps)

    ind_proposal = independent_proposal((; t=t, r=r), z)
    ind_mh_kernel = MhKernel(rng, ind_proposal)
    ind_smc_mh = SequentialMonteCarlo(ind_mh_kernel, temp_schedule, params.n_particles, log(params.relative_ess * params.n_particles))

    t_sym = BroadcastedNode(:t, rng, KernelNormal, 0, params.proposal_σ_t)
    r_sym = BroadcastedNode(:r, rng, QuaternionPerturbation, params.proposal_σ_r_quat)
    sym_proposal = symmetric_proposal((; t=t_sym, r=r_sym), z)

    sym_fp_kernel = ForwardProposalKernel(sym_proposal)
    sym_smc_fp = SequentialMonteCarlo(sym_fp_kernel, temp_schedule, params.n_particles, log(params.relative_ess * params.n_particles))

    sym_mh_kernel = MhKernel(rng, sym_proposal)
    sym_smc_mh = SequentialMonteCarlo(sym_mh_kernel, temp_schedule, params.n_particles, log(params.relative_ess * params.n_particles))

    sym_boot_kernel = BootstrapKernel(sym_proposal)
    sym_smc_boot = SequentialMonteCarlo(sym_boot_kernel, temp_schedule, params.n_particles, log(params.relative_ess * params.n_particles))

    # NOTE ind_smc only makes sense when using a MCMCKernel, otherwise I throw away all the information
    composed_sampler = ComposedSampler(Weights([0.1, 1.0]), ind_smc_mh, sym_smc_mh)

    sampler = composed_sampler
    # sampler = sym_smc_mh
    # sampler = sym_smc_fp
    # NOTE tends to diverge with to few samples, since there is no prior pulling it back to sensible values. But it can also converge to very precise values since there is no prior holding it back.
    # sampler = sym_smc_boot

    sample, state = step(rng, posterior, sampler)
    @progress for _ in 1:params.n_steps
        sample, state = step(rng, posterior, sampler, state)
    end
    sample, state
end

# NOTE SMC: tempering is essential. More steps (MCMC) allows higher normalization_constant than more particles (FP, Bootstrap), 15-30 seems to be a good range
@reset parameters.normalization_constant = 25;
@reset parameters.proposal_σ_r_quat = 0.1;
@reset parameters.proposal_σ_t = [0.01, 0.01, 0.01];
@reset parameters.seed = rand(RandomDevice(), UInt32);
# NOTE resampling dominated like FP & Bootstrap kernels typically perform better with more samples (1_000,100) while MCMC kernels tend to perform better with more steps (2_000,50)
@reset parameters.n_steps = 2_500
@reset parameters.n_particles = 100
# Normalization and tempering leads to less resampling, especially in MCMC sampler
@reset parameters.relative_ess = 0.8;
final_sample, final_state = run_inference(gl_context, parameters, experiment);

println("Final log-evidence: $(final_state.log_evidence)")
plot_pose_density(final_sample; trim=false)

anim = @animate for i ∈ 0:2:360
    scatter_position(final_sample, 100, label="particle number", camera=(i, 25), projection_type=:perspective, legend_position=:topright)
end;
gif(anim, "anim_fps15.gif", fps=20)
# TODO diagnostics: Accepted steps, resampling steps
