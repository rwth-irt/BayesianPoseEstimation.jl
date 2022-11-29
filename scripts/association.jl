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
parameters = @set parameters.device = :CUDA
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

function run_inference(parameters::Parameters, render_context, observation; kwargs...)
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
    r = BroadcastedNode(:r, rng, QuaternionDistribution, parameters.precision)

    μ_fn = render_fn | (render_context, Scene(parameters, render_context), parameters.object_id)
    μ = DeterministicNode(:μ, μ_fn, (; t=t, r=r))

    # NOTE the σ of the association must be larger than the one for the pose estimation
    dist_is = valid_pixel_normal | parameters.association_σ
    dist_not = valid_pixel_tail | (parameters.min_depth, parameters.max_depth, parameters.pixel_θ)
    association_fn = pixel_association | (dist_is, dist_not, parameters.prior_o)
    o = DeterministicNode(:o, (expectation) -> association_fn.(expectation, observation.z), (; μ=μ))

    # NOTE valid_pixel diverges without normalization
    pixel_model = valid_pixel_mixture | (parameters.min_depth, parameters.max_depth, parameters.pixel_θ, parameters.pixel_σ)

    # NOTE normalization does not seem to be required or is even worse
    z = BroadcastedNode(:z, dev_rng, pixel_model, (; μ=μ, o=o))
    z_norm = ModifierNode(z, dev_rng, ImageLikelihoodNormalizer | parameters.normalization_constant)

    posterior = PosteriorModel(z_norm, observation)

    # Assemble samplers
    # t & r change expected depth, o not
    t_ind = IndependentProposal(t, z)
    t_ind_mh = MetropolisHastings(t_ind)

    t_sym = SymmetricProposal(BroadcastedNode(:t, rng, KernelNormal, 0, parameters.proposal_σ_t), z)
    t_sym_mh = MetropolisHastings(t_sym)

    r_ind = IndependentProposal(r, z)
    r_ind_mh = MetropolisHastings(r_ind)

    r_sym = QuaternionProposal(BroadcastedNode(:r, rng, QuaternionPerturbation, parameters.proposal_σ_r_quat), z)
    r_sym_mh = MetropolisHastings(r_sym)

    # ComposedSampler
    # NOTE Independent should have low weights because almost no samples will be accepted
    composed_sampler = ComposedSampler(Weights([0.1, 0.01, 0.1, 1.0]), t_ind_mh, r_ind_mh, t_sym_mh, r_sym_mh)

    # WARN random acceptance needs to be calculated on CPU, thus CPU rng
    chain = sample(rng, posterior, composed_sampler, 10_000; discard_initial=0_000, thinning=1, kwargs...)

    map(chain) do sample
        s, _ = to_model_domain(sample, bijector(z))
        s
    end
end

observation = fake_observation(parameters, render_context, 0.4)
# plot_depth_img(Array(obs.z))
parameters = @set parameters.seed = rand(RandomDevice(), UInt32)
model_chain = run_inference(parameters, render_context, observation; discard_initial=0_000, thinning=2);
# NOTE looks like sampling a pole which is probably sampling uniformly and transforming it back to Euler
plot_pose_chain(model_chain)
plot_logprob(model_chain, 200)

# NOTE This does not look too bad. The most likely issue is the logjac correction which is calculated over all the pixels instead of the valid
plot_prob_img(mean_image(model_chain, :o) |> Array)
plot_prob_img(model_chain[end].variables.o |> Array)

gr()
anim = @animate for i ∈ 0:2:360
    scatter_position(model_chain, 100, camera=(i, 25), projection_type=:perspective, legend_position=:topright)
end
gif(anim, "anim_fps15.gif", fps=20)
pyplot()