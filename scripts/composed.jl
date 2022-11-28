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
    # nominal scene
    obs_params = @set parameters.mesh_files = ["meshes/monkey.obj", "meshes/cube.obj", "meshes/cube.obj"]
    obs_scene = Scene(obs_params, render_context)
    obs_scene = @set obs_scene.meshes[2].pose.translation = Translation(0.1, 0, 3)
    obs_scene = @set obs_scene.meshes[2].scale = Scale(1.8, 1.5, 1)
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

function plot_o_chain(model_chain, step=200)
    plt_o_chain = plot_variable(model_chain, :o, step; label="o", xlabel="Iteration [÷ $(step)]", ylabel="Zugehörigkeit", legend=false)
    plt_o_dens = density_variable(model_chain, :o; label="o", xlabel="Zugehörigkeit [0,1]", ylabel="Wahrscheinlichkeit", legend=false)
    plot(plt_o_chain, plt_o_dens)
end

function run_inference(parameters::Parameters, render_context, obs; kwargs...)
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
    # NOTE takes longer to converge than Dirac and does not work well if actual occlusion is present. Occlusion should lead to low o which leads to low confidence in the data
    o = BroadcastedNode(:o, dev_rng, Dirac, parameters.prior_o)
    # TODO o = BroadcastedNode(:o, dev_rng, KernelUniform, parameters.precision)

    μ_fn = render_fn | (render_context, Scene(parameters, render_context), parameters.object_id)
    μ = DeterministicNode(:μ, μ_fn, (; t=t, r=r))

    # Depth image model
    # Explicitly handles invalid μ → no normalization
    # NOTE Using the actual number of pixels makes the model overconfident due to the seemingly large amount of data compared to the prior. Make this adaptive or formalize it?
    # norm_const = expected_pixel_count(rng, prior_model, render_context, scene, parameters)
    # NOTE ValidPixel is required for occlusions? Combined with pixel_explicit even better?
    pixel_model = valid_pixel_mixture | (parameters.min_depth, parameters.max_depth, parameters.pixel_θ, parameters.pixel_σ)
    z = BroadcastedNode(:z, dev_rng, pixel_model, (; μ=μ, o=o))
    z_norm = ModifierNode(z, dev_rng, ImageLikelihoodNormalizer | parameters.normalization_constant)

    # NOTE normalized_posterior seems way better but is a bit slower. Occlusions: Diverges to the occluding object for z_norm. ValidPixel Diverges to max_depth without normalization
    posterior = PosteriorModel(z_norm, obs)

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

    o_ind = IndependentProposal(o, z)
    o_ind_mh = MetropolisHastings(o_ind)

    o_sym = SymmetricProposal(BroadcastedNode(:o, dev_rng, KernelNormal, 0, parameters.proposal_σ_o), z)
    o_sym_mh = MetropolisHastings(o_sym)

    # ComposedSamplers
    # NOTE sampling scalar o is relatively cheap but struggles when occluded
    ind_mh = ComposedSampler(t_ind_mh, r_ind_mh, o_ind_mh)
    # NOTE works better than combination? Maybe because we do not sample the poles like with RotXYZ
    sym_mh = ComposedSampler(t_sym_mh, r_sym_mh, o_sym_mh)

    # ind_sym = ComposedSampler(Weights([0.1, 0.1, 0.1, 0.1, 1.0, 0.1]), t_ind_mh, r_ind_mh, o_ind_mh, t_sym_mh, r_sym_mh, o_sym_mh)
    # NOTE Sampling mostly r_sym_mh seems to converge faster?
    ind_sym = ComposedSampler(Weights([0.1, 0.1, 0.1, 1.0]), t_ind_mh, r_ind_mh, t_sym_mh, r_sym_mh)

    # WARN random acceptance needs to be calculated on CPU, thus CPU rng
    chain = sample(rng, posterior, ind_sym, 10_000; discard_initial=0_000, thinning=1, kwargs...)

    map(chain) do sample
        s, _ = to_model_domain(sample, bijector(z))
        s
    end
end

obs = fake_observation(parameters, render_context, 0.4)

parameters = @set parameters
# NOTE optimal parameter values seem to be inversely correlated
parameters = @set parameters.pixel_σ = 0.01
parameters = @set parameters.normalization_constant = 15
model_chain = run_inference(parameters, render_context, obs; thinning=2);
plot_pose_chain(model_chain)

plot_logprob(model_chain, 200)
# NOTE I would have expected it to converge around 0.8
plot_o_chain(model_chain, 200)

# gr()
# anim = @animate for i ∈ 0:2:360
#     scatter_position(model_chain, 100, camera=(i, 25), projection_type=:perspective, legend_position=:outertop)
# end
# gif(anim, "anim_fps15.gif", fps=20)
# pyplot()

# plot(
#     plot_depth_img(obs.z |> Array),
#     plot_depth_img(render_fn(render_context, Scene(parameters, render_context), 1, model_chain[end].variables.t, model_chain[end].variables.r) |> Array))
