# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using Accessors
using CUDA
using MCMCDepth
using Random
using Plots

gr()
MCMCDepth.diss_defaults()

parameters = Parameters()
parameters = @set parameters.device = :CUDA
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

function run_inference(parameters::Parameters, render_context, observation, n_steps=1_000, n_tries=250; kwargs...)
    # Device
    if parameters.device === :CUDA
        CUDA.allowscalar(false)
    end
    # RNGs
    rng = cpu_rng(parameters)
    dev_rng = device_rng(parameters)

    # Model specification
    # Pose must be calculated on CPU since there is now way to pass it from CUDA to OpenGL
    t = BroadcastedNode(:t, rng, KernelNormal, parameters.mean_t, parameters.σ_t)
    r = BroadcastedNode(:r, rng, QuaternionUniform, parameters.precision)

    μ_fn = render_fn | (render_context, Scene(parameters, render_context), parameters.object_id)
    μ = DeterministicNode(:μ, μ_fn, (; t=t, r=r))

    # NOTE Analytic pixel association is only a deterministic function and not a Gibbs sampler in the traditional sense. Gibbs sampler would call rand(q(o|t,r,μ)) and not fn(μ,z). Probably "collapsed Gibbs" is the correct expression for it.
    # NOTE the σ of the association must be larger than the one for the pose estimation
    dist_is = valid_pixel_normal | parameters.association_σ
    dist_not = smooth_valid_tail | (parameters.min_depth, parameters.max_depth, parameters.pixel_θ, parameters.association_σ)
    association_fn = pixel_association | (dist_is, dist_not, parameters.prior_o)
    o = DeterministicNode(:o, (expectation) -> association_fn.(expectation, observation.z), (; μ=μ))

    # NOTE valid_pixel diverges without normalization
    pixel_model = smooth_valid_mixture | (parameters.min_depth, parameters.max_depth, parameters.pixel_θ, parameters.pixel_σ)

    # NOTE normalization does not seem to be required or is even worse
    z = BroadcastedNode(:z, dev_rng, pixel_model, (; μ=μ, o=o))
    z_norm = ModifierNode(z, dev_rng, ImageLikelihoodNormalizer | parameters.normalization_constant)

    posterior = PosteriorModel(z_norm, observation)

    # Assemble samplers
    # t & r change expected depth, o not
    t_ind = independent_proposal(t, z)
    t_ind_mh = MetropolisHastings(t_ind)
    # NOTE twice the compute budget
    t_ind_mtm = MultipleTry(t_ind, n_tries * 2)

    t_sym = symmetric_proposal(BroadcastedNode(:t, rng, KernelNormal, 0, parameters.proposal_σ_t), z)
    t_sym_mh = MetropolisHastings(t_sym)

    t_add = additive_proposal(BroadcastedNode(:t, rng, KernelNormal, 0, parameters.proposal_σ_t), z)
    t_add_mtm = MultipleTry(t_add, n_tries)

    r_ind = independent_proposal(r, z)
    r_ind_mh = MetropolisHastings(r_ind)
    # NOTE twice the compute budget
    r_ind_mtm = MultipleTry(r_ind, n_tries * 2)

    r_sym = symmetric_proposal(BroadcastedNode(:r, rng, QuaternionPerturbation, parameters.proposal_σ_r_quat), z)
    r_sym_mh = MetropolisHastings(r_sym)

    r_add = additive_proposal(BroadcastedNode(:r, rng, QuaternionPerturbation, parameters.proposal_σ_r_quat), z)
    r_add_mtm = MultipleTry(r_add, n_tries)

    # ComposedSampler
    # NOTE Independent should have low weights because almost no samples will be accepted
    # NOTE These parameters seem to be quite important for convergence, especially r_ind_mh ≪ r_sym_mh
    composed_sampler = ComposedSampler(Weights([0.1, 0.1, 1.0, 1.0]), t_ind_mtm, r_ind_mtm, t_add_mtm, r_add_mtm)
    # composed_sampler = ComposedSampler(Weights([0.1, 1.0, 0.1, 1.0]), t_ind_mh, r_ind_mh, t_sym_mh, r_sym_mh)

    # WARN random acceptance needs to be calculated on CPU, thus CPU rng
    chain = sample(rng, posterior, composed_sampler, n_steps; discard_initial=0_000, thinning=1, kwargs...)

    map(chain) do sample
        s, _ = to_model_domain(sample, bijector(z))
        s
    end
end

# plot_depth_img(Array(obs.z))
# NOTE optimal parameter values of pixel_σ and normalization_constant seem to be inversely correlated. Moreover, different values seem to be optimal when using analytic association
parameters = @set parameters.normalization_constant = 20
# NOTE Should be able to increase σ in MTM
parameters = @set parameters.proposal_σ_r_quat = 0.3
parameters = @set parameters.proposal_σ_t = [0.02, 0.02, 0.02]
parameters = @set parameters.seed = rand(RandomDevice(), UInt32)
model_chain = run_inference(parameters, gl_context, observation, 2_000, 25; thinning=1);
# NOTE looks like sampling a pole which is probably sampling uniformly and transforming it back to Euler
plot_pose_chain(model_chain, 50)
plot_logprob(model_chain, 50)

# NOTE This does not look too bad. The most likely issue is the logjac correction which is calculated over all the pixels instead of the valid
plot_prob_img(mean_image(model_chain, :o) |> Array)
plot_prob_img(model_chain[end].variables.o |> Array)

anim = @animate for i ∈ 0:2:360
    scatter_position(model_chain; camera=(i, 25), projection_type=:perspective, legend_position=:topright)
end;
gif(anim, "anim_fps15.gif", fps=20)
