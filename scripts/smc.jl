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

CUDA.allowscalar(false)
gr()
MCMCDepth.diss_defaults()

parameters = Parameters()
# NOTE SMC: tempering is essential. More steps (MCMC) allows higher normalization_constant than more particles (FP, Bootstrap), 15-30 seems to be a good range
@reset parameters.normalization_constant = 25;
@reset parameters.proposal_σ_r_quat = 0.1;
@reset parameters.proposal_σ_t = [0.01, 0.01, 0.01];
# TODO move to top
@reset parameters.seed = rand(RandomDevice(), UInt32);
# NOTE resampling dominated like FP & Bootstrap kernels typically perform better with more samples (1_000,100) while MCMC kernels tend to perform better with more steps (2_000,50)
@reset parameters.n_steps = 1_000
@reset parameters.n_particles = 50
# Normalization and tempering leads to less resampling, especially in MCMC sampler
@reset parameters.relative_ess = 0.8;

# NOTE takes minutes instead of seconds
# @reset parameters.device = :CPU
cpu_rng = rng(parameters)
dev_rng = device_rng(parameters)
gl_context = render_context(parameters)

include("fake_observation.jl")
camera = fake_camera(parameters)
model = upload_mesh(gl_context, fake_mesh)
prior_t = fake_gt_position + [0.05, -0.05, -0.1]
fake_img = fake_observation(gl_context, parameters, 0.4)

experiment = Experiment(Scene(camera, [model]), prior_t, fake_img)

function posterior_model(gl_context, params, experiment, rng, dev_rng)
    t = BroadcastedNode(:t, rng, KernelNormal, experiment.prior_t, params.σ_t)
    r = BroadcastedNode(:r, rng, QuaternionUniform, params.float_type)

    μ_fn = render_fn | (gl_context, experiment.scene)
    μ = DeterministicNode(:μ, μ_fn, (; t=t, r=r))

    o_fn = smooth_association_fn(params)
    # condition on data via closure
    o = DeterministicNode(:o, μ -> o_fn.(μ, experiment.depth_image), (; μ=μ))
    # NOTE almost no performance gain over DeterministicNode
    # o = BroadcastedNode(:o, dev_rng, KernelDirac, parameters.prior_o)

    # NOTE valid_pixel diverges without normalization
    pixel_model = smooth_valid_mixture | (params.min_depth, params.max_depth, params.pixel_θ, params.pixel_σ)
    z = BroadcastedNode(:z, dev_rng, pixel_model, (; μ=μ, o=o))
    z_norm = ModifierNode(z, dev_rng, ImageLikelihoodNormalizer | params.normalization_constant)

    PosteriorModel(z_norm, (; z=experiment.depth_image))
end
posterior = posterior_model(gl_context, parameters, experiment, cpu_rng, dev_rng)

function smc_forward(rng, params, posterior)
    temp_schedule = LinearSchedule(params.n_steps)

    # NOTE use independent proposals only with an MCMC Kernel, otherwise all information is thrown away.
    t_sym = BroadcastedNode(:t, rng, KernelNormal, 0, params.proposal_σ_t)
    r_sym = BroadcastedNode(:r, rng, QuaternionPerturbation, params.proposal_σ_r_quat)
    t_sym_proposal = symmetric_proposal((; t=t_sym), posterior.node)
    r_sym_proposal = symmetric_proposal((; r=r_sym), posterior.node)
    proposals = (t_sym_proposal, r_sym_proposal)
    weights = Weights([1.0, 1.0])

    samplers = map(proposals) do proposal
        mh_kernel = ForwardProposalKernel(proposal)
        SequentialMonteCarlo(mh_kernel, temp_schedule, params.n_particles, log(params.relative_ess * params.n_particles))
    end
    ComposedSampler(weights, samplers...)
end

# NOTE tends to diverge with to few samples, since there is no prior pulling it back to sensible values. But it can also converge to very precise values since there is no prior holding it back.
function smc_bootstrap(rng, params, posterior)
    temp_schedule = LinearSchedule(params.n_steps)

    # NOTE use independent proposals only with an MCMC Kernel, otherwise all information is thrown away.
    t_sym = BroadcastedNode(:t, rng, KernelNormal, 0, params.proposal_σ_t)
    r_sym = BroadcastedNode(:r, rng, QuaternionPerturbation, params.proposal_σ_r_quat)
    t_sym_proposal = symmetric_proposal((; t=t_sym), posterior.node)
    r_sym_proposal = symmetric_proposal((; r=r_sym), posterior.node)
    proposals = (t_sym_proposal, r_sym_proposal)
    weights = Weights([1.0, 1.0])

    samplers = map(proposals) do proposal
        mh_kernel = BootstrapKernel(proposal)
        SequentialMonteCarlo(mh_kernel, temp_schedule, params.n_particles, log(params.relative_ess * params.n_particles))
    end
    ComposedSampler(weights, samplers...)
end

function smc_mh(rng, params, posterior)
    # NOTE LinearSchedule seems reasonable, ExponentialSchedule and ConstantSchedule either explore too much or not enough
    temp_schedule = LinearSchedule(params.n_steps)

    t_ind = BroadcastedNode(:t, rng, KernelNormal, experiment.prior_t, params.σ_t)
    r_ind = BroadcastedNode(:r, rng, QuaternionUniform, params.float_type)
    t_ind_proposal = independent_proposal((; t=t_ind), posterior.node)
    r_ind_proposal = independent_proposal((; r=r_ind), posterior.node)

    t_sym = BroadcastedNode(:t, rng, KernelNormal, 0, params.proposal_σ_t)
    r_sym = BroadcastedNode(:r, rng, QuaternionPerturbation, params.proposal_σ_r_quat)
    t_sym_proposal = symmetric_proposal((; t=t_sym), posterior.node)
    r_sym_proposal = symmetric_proposal((; r=r_sym), posterior.node)
    proposals = (t_sym_proposal, r_sym_proposal, t_ind_proposal, r_ind_proposal)
    weights = Weights([1.0, 1.0, 0.1, 0.1])

    samplers = map(proposals) do proposal
        mh_kernel = MhKernel(rng, proposal)
        SequentialMonteCarlo(mh_kernel, temp_schedule, params.n_particles, log(params.relative_ess * params.n_particles))
    end

    # TODO is Gibbs for t & r valid in SMC?
    ComposedSampler(weights, samplers...)
end

# TODO implement for AbstractSampler? specialize sample?
function run_inference(rng, posterior, sampler, params::Parameters)
    sample, state = step(rng, posterior, sampler)
    @progress for _ in 1:params.n_steps
        sample, state = step(rng, posterior, sampler, state)
    end
    sample, state
end

sampler = smc_mh(cpu_rng, parameters, posterior)
# sampler = smc_bootstrap(cpu_rng, parameters, posterior)
# sampler = smc_forward(cpu_rng, parameters, posterior)

@reset parameters.n_steps = 1_000
final_sample, final_state = run_inference(cpu_rng, posterior, sampler, parameters);
println("Final log-evidence: $(final_state.log_evidence)")
plot_pose_density(final_sample; trim=false)

anim = @animate for i ∈ 0:2:360
    scatter_position(final_sample, 100, label="particle number", camera=(i, 25), projection_type=:perspective, legend_position=:topright)
end;
gif(anim, "anim_fps15.gif", fps=20)
# TODO diagnostics: Accepted steps, resampling steps
