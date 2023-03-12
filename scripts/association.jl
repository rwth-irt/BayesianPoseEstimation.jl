# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

using Accessors
using CUDA
using MCMCDepth
using Random
using Plots

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
# TODO Sometimes the scene is not rendered?
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
    # Pose must be calculated on CPU since there is now way to pass it from CUDA to OpenGL
    t = BroadcastedNode(:t, rng, KernelNormal, experiment.prior_t, params.σ_t)
    r = BroadcastedNode(:r, rng, QuaternionUniform, params.float_type)

    μ_fn = render_fn | (render_context, experiment.scene)
    μ = DeterministicNode(:μ, μ_fn, (; t=t, r=r))

    # NOTE Analytic pixel association is only a deterministic function and not a Gibbs sampler in the traditional sense. Gibbs sampler would call rand(q(o|t,r,μ)) and not fn(μ,z). Probably "collapsed Gibbs" is the correct expression for it.
    # NOTE the σ of the association must be larger than the one for the pose estimation
    dist_is = valid_pixel_normal | params.association_σ
    dist_not = smooth_valid_tail | (params.min_depth, params.max_depth, params.pixel_θ, params.association_σ)
    association_fn = pixel_association | (dist_is, dist_not, params.prior_o)
    o = DeterministicNode(:o, (expectation) -> association_fn.(expectation, experiment.depth_image), (; μ=μ))

    # NOTE valid_pixel diverges without normalization
    pixel_model = smooth_valid_mixture | (params.min_depth, params.max_depth, params.pixel_θ, params.pixel_σ)

    # NOTE normalization does not seem to be required or is even worse
    z = BroadcastedNode(:z, dev_rng, pixel_model, (; μ=μ, o=o))
    z_norm = ModifierNode(z, dev_rng, ImageLikelihoodNormalizer | params.normalization_constant)

    posterior = PosteriorModel(z_norm, (; z=experiment.depth_image))

    # Assemble samplers
    # t & r change expected depth, o not
    t_ind = independent_proposal(t, z)
    t_ind_mh = MetropolisHastings(t_ind)
    # NOTE twice the compute budget
    t_ind_mtm = MultipleTry(t_ind, params.n_particles * 2)

    t_sym = symmetric_proposal(BroadcastedNode(:t, rng, KernelNormal, 0, params.proposal_σ_t), z)
    t_sym_mh = MetropolisHastings(t_sym)

    t_add = additive_proposal(BroadcastedNode(:t, rng, KernelNormal, 0, params.proposal_σ_t), z)
    t_add_mtm = MultipleTry(t_add, params.n_particles)

    r_ind = independent_proposal(r, z)
    r_ind_mh = MetropolisHastings(r_ind)
    # NOTE twice the compute budget
    r_ind_mtm = MultipleTry(r_ind, params.n_particles * 2)

    r_sym = symmetric_proposal(BroadcastedNode(:r, rng, QuaternionPerturbation, params.proposal_σ_r_quat), z)
    r_sym_mh = MetropolisHastings(r_sym)

    r_add = additive_proposal(BroadcastedNode(:r, rng, QuaternionPerturbation, params.proposal_σ_r_quat), z)
    r_add_mtm = MultipleTry(r_add, params.n_particles)

    # ComposedSampler
    # NOTE Independent should have low weights because almost no samples will be accepted
    # NOTE These parameters seem to be quite important for convergence
    composed_sampler = ComposedSampler(Weights([0.1, 0.1, 1.0, 1.0]), t_ind_mtm, r_ind_mtm, t_add_mtm, r_add_mtm)
    # composed_sampler = ComposedSampler(Weights([0.1, 0.1, 1.0, 1.0]), t_ind_mh, r_ind_mh, t_sym_mh, r_sym_mh)

    # WARN random acceptance needs to be calculated on CPU, thus CPU rng
    chain = sample(rng, posterior, composed_sampler, params.n_steps; discard_initial=params.n_burn_in, thinning=params.n_thinning, kwargs...)

    map(chain) do sample
        s, _ = to_model_domain(sample, bijector(z))
        s
    end
end

# plot_depth_img(Array(obs.z))
# NOTE optimal parameter values of pixel_σ and normalization_constant seem to be inversely correlated. Moreover, different values seem to be optimal when using analytic association
@reset parameters.normalization_constant = 25
# NOTE Should be able to increase σ in MTM
@reset parameters.proposal_σ_r_quat = 0.3
@reset parameters.proposal_σ_t = [0.02, 0.02, 0.02]
@reset parameters.seed = rand(RandomDevice(), UInt32)
@reset parameters.n_steps = 1_000
@reset parameters.n_burn_in = 0
@reset parameters.n_thinning = 1
@reset parameters.n_particles = 150
model_chain = run_inference(gl_context, parameters, experiment);
# NOTE looks like sampling a pole which is probably sampling uniformly and transforming it back to Euler
plot_pose_chain(model_chain, 50)
plot_logprob(model_chain, 50)

# NOTE This does not look too bad. The most likely issue is the logjac correction which is calculated over all the pixels instead of the valid
plot_prob_img(mean_image(model_chain, :o))
plot_prob_img(model_chain[end].variables.o)

anim = @animate for i ∈ 0:2:360
    scatter_position(model_chain; camera=(i, 25), projection_type=:perspective, legend_position=:topright)
end;
gif(anim, "anim_fps15.gif", fps=20)
