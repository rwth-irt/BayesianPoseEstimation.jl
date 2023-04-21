# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using CUDA
using MCMCDepth
using Random
using Plots

CUDA.allowscalar(false)
gr()
MCMCDepth.diss_defaults()

parameters = Parameters()
# NOTE optimal parameter values of pixel_σ and normalization_constant seem to be inversely correlated. Moreover, different values seem to be optimal when using analytic association
@reset parameters.normalization_constant = 25
# NOTE Should be able to increase σ in MTM
@reset parameters.proposal_σ_r_quat = 0.5
@reset parameters.proposal_σ_t = [0.02, 0.02, 0.02]
# TODO same seed for experiments
@reset parameters.seed = rand(RandomDevice(), UInt32)
@reset parameters.n_steps = 1_000
@reset parameters.n_burn_in = 0
@reset parameters.n_thinning = 1
@reset parameters.n_particles = 100
# TODO tempering in MCMC?

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

# TODO find a common place?
function posterior_model(gl_context, params, experiment, rng, dev_rng)
    t = BroadcastedNode(:t, rng, KernelNormal, experiment.prior_t, params.σ_t)
    r = BroadcastedNode(:r, rng, QuaternionUniform, params.float_type)

    μ_fn = render_fn | (gl_context, experiment.scene)
    μ = DeterministicNode(:μ, μ_fn, (; t=t, r=r))

    # NOTE Analytic pixel association is only a deterministic function and not a Gibbs sampler in the traditional sense. Gibbs sampler would call rand(q(o|t,r,μ)) and not fn(μ,z). Probably "collapsed Gibbs" is the correct expression for it.
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

function mh_sampler(rng, params, posterior)
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
        MetropolisHastings(proposal)
    end
    ComposedSampler(weights, samplers...)
end

function mtm_sampler(rng, params, posterior)
    t_ind = BroadcastedNode(:t, rng, KernelNormal, experiment.prior_t, params.σ_t)
    r_ind = BroadcastedNode(:r, rng, QuaternionUniform, params.float_type)
    t_ind_proposal = independent_proposal((; t=t_ind), posterior.node)
    r_ind_proposal = independent_proposal((; r=r_ind), posterior.node)

    t_sym = BroadcastedNode(:t, rng, KernelNormal, 0, params.proposal_σ_t)
    r_sym = BroadcastedNode(:r, rng, QuaternionPerturbation, params.proposal_σ_r_quat)
    # TODO additive required for MTM? Probably because transition probability would be scalar?
    t_sym_proposal = symmetric_proposal((; t=t_sym), posterior.node)
    r_sym_proposal = symmetric_proposal((; r=r_sym), posterior.node)

    proposals = (t_sym_proposal, r_sym_proposal, t_ind_proposal, r_ind_proposal)
    weights = Weights([1.0, 1.0, 0.01, 0.01])
    # proposals = (t_ind_proposal, r_ind_proposal)
    # proposals = (t_sym_proposal, r_sym_proposal)
    # weights = Weights([1.0, 1.0])

    samplers = map(proposals) do proposal
        MultipleTry(proposal, params.n_particles)
    end
    ComposedSampler(weights, samplers...)
end

# sampler = mh_sampler(cpu_rng, parameters, posterior)
sampler = mtm_sampler(cpu_rng, parameters, posterior)

chain = sample(cpu_rng, posterior, sampler, parameters.n_steps; discard_initial=parameters.n_burn_in, thinning=parameters.n_thinning);
# TODO no conversion in smc? Also see next NOTE
# NOTE looks like sampling a pole which is probably sampling uniformly and transforming it back to Euler
plot_pose_chain(chain, 50)
plot_logprob(chain, 50)

# NOTE This does not look too bad. The most likely issue is the logjac correction which is calculated over all the pixels instead of the valid
plot_prob_img(mean_image(chain, :o))
plot_prob_img(chain[end].variables.o)

anim = @animate for i ∈ 0:2:360
    scatter_position(chain; camera=(i, 25), projection_type=:perspective, legend_position=:topright)
end;
gif(anim, "anim_fps15.gif", fps=20)
