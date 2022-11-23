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

# TODO Do I want a main method with a plethora of parameters?
# https://bkamins.github.io/julialang/2022/07/15/main.html
# TODO these are experiment specific design decision
parameters = Parameters()
parameters = @set parameters.device = :CUDA

# Device
if parameters.device === :CUDA
    CUDA.allowscalar(false)
end

# RNGs
rng = cpu_rng(parameters)
dev_rng = device_rng(parameters)
# Run specific parts on the GPU
dev_model = RngModel | dev_rng

# Render context
render_context = RenderContext(parameters.width, parameters.height, parameters.depth, device_array_type(parameters))
scene = Scene(parameters, render_context)
render_model = RenderModel | (render_context, scene, parameters.object_id)

# Model specification
# TODO lot of copy - paste. Function to generate samplers?
# Pose must be calculated on CPU since there is now way to pass it from CUDA to OpenGL
t = BroadcastedNode(:t, rng, KernelNormal, parameters.mean_t, parameters.σ_t)
r = BroadcastedNode(:r, rng, QuaternionDistribution, parameters.precision)
# TODO Show robustness for different o by estimating it
# o = BroadcastedNode(:o, rng, Dirac, parameters.precision(3 / 4))
# TODO works but takes longer to converge
o = BroadcastedNode(:o, dev_rng, KernelUniform, parameters.precision)

function render_fn(render_context, scene, object_id, t, r)
    p = to_pose(t, r)
    render(render_context, scene, object_id, p)
end
μ_fn = render_fn | (render_context, scene, parameters.object_id)
μ = DeterministicNode(:μ, μ_fn, (; t=t, r=r))

# Depth image model
# Does not handle invalid μ → requires normalization
pixel_mix = pixel_mixture(parameters)
# Explicitly handles invalid μ → no normalization
pixel_expl = pixel_explicit(parameters)
z = BroadcastedNode(:z, dev_rng, pixel_mix, (; μ=μ, o=o))
# TODO Using the actual number of pixels makes the model overconfident due to the seemingly large amount of data compared to the prior. Make this adaptive or formalize it?
# norm_const = expected_pixel_count(rng, prior_model, render_context, scene, parameters)
z_norm = ModifierNode(z, dev_rng, ImageLikelihoodNormalizer | parameters.precision(10))
z_seq = sequentialize(z_norm)
s = rand(z_seq, 2);
plot(plot_depth_img(Array(s.μ[:, :, 1])), plot_prob_img(Array(s.o[:, :, 1])), plot_depth_img(Array(s.z[:, :, 1])),
    plot_depth_img(Array(s.μ[:, :, 2])), plot_prob_img(Array(s.o[:, :, 2])), plot_depth_img(Array(s.z[:, :, 2])))

# Assemble samplers
# t & r change expected depth, o not
t_ind = IndependentProposal(t, z_norm)
t_ind_mh = MetropolisHastings(t_ind)

t_sym = SymmetricProposal(BroadcastedNode(:t, rng, KernelNormal, 0, parameters.proposal_σ_t), z_norm)
t_sym_mh = MetropolisHastings(t_sym)

r_ind = IndependentProposal(r, z_norm)
r_ind_mh = MetropolisHastings(r_ind)

r_sym = QuaternionProposal(BroadcastedNode(:r, rng, QuaternionPerturbation, parameters.proposal_σ_r_quat), z_norm)
r_sym_mh = MetropolisHastings(r_sym)

o_ind = IndependentProposal(o, z_norm)
o_ind_mh = MetropolisHastings(o_ind)

o_sym = SymmetricProposal(BroadcastedNode(:o, dev_rng, KernelNormal, 0, parameters.proposal_σ_o), z_norm)
o_sym_mh = MetropolisHastings(o_sym)

# ComposedSamplers
ind_sym_sampler = ComposedSampler(Weights([0.1, 0.1, 0.1, 1.0, 1.0, 1.0]), t_ind_mh, r_ind_mh, o_ind_mh, t_sym_mh, r_sym_mh, o_sym_mh)
ind_sampler = ComposedSampler(t_ind_mh, r_ind_mh, o_ind_mh)
# TODO works better than combination? Maybe because we do not sample the poles like with RotXYZ
sym_sampler = ComposedSampler(t_sym_mh, r_sym_mh, o_sym_mh)


# TODO the interface sucks
# dist_is(σ, μ) = ValidPixel(μ, KernelNormal(μ, σ))
# dist_not(μ) = ValidPixel(μ, pixel_tail | (parameters.min_depth, parameters.max_depth, parameters.pixel_θ))
# o_association = ImageAssociation | (dist_is | parameters.pixel_σ, dist_not | parameters.pixel_θ, parameters.prior_o)
# o_analytic(s::Sample) = o_association((variables(s).μ), variables(s).z)
# o_gibbs = Gibbs(prior_model, o_analytic)

# PosteriorModel

# Fake observation
# TODO occlusion
obs_params = @set parameters.mesh_files = ["meshes/monkey.obj", "meshes/cube.obj"]
obs_scene = Scene(obs_params, render_context)
obs_scene = @set obs_scene.meshes[2].pose.translation = Translation(0.1, 0, 3)
obs_scene = @set obs_scene.meshes[2].scale = Scale(1.8, 1.5, 1)
obs_μ = render(render_context, obs_scene, parameters.object_id, to_pose(parameters.mean_t + [0.05, -0.05, -0.1], [0, 0, 0]))
obs = rand(z, (; μ=obs_μ, o=0.8f0))[(:z,)]

# TODO normalized_posterior seems way better but is a bit slower
posterior = PosteriorModel(z_norm, obs)


# WARN random acceptance needs to be calculated on CPU, thus CPU rng
chain = sample(rng, posterior, ind_sym_sampler, 20_000; discard_initial=0_000, thinning=2);

# TODO separate evaluation from experiments, i.e. save & load
model_chain = map(chain) do sample
    s, _ = to_model_domain(sample, bijector(z_norm))
    s
end;
STEP = 300

plt_t_chain = plot_variable(model_chain, :t, STEP; label=["x" "y" "z"], xlabel="Iteration [÷ $(STEP)]", ylabel="Position [m]", legend=false);
plt_t_dens = density_variable(model_chain, :t; label=["x" "y" "z"], xlabel="Position [m]", ylabel="Wahrscheinlichkeit", legend=:outerright, left_margin=5mm);

plt_r_chain = plot_variable(model_chain, :r, STEP; label=["x" "y" "z"], xlabel="Iteration [÷ $(STEP)]", ylabel="Orientierung [rad]", legend=false, top_margin=5mm);
plt_r_dens = density_variable(model_chain, :r; label=["x" "y" "z"], xlabel="Orientierung [rad]", ylabel="Wahrscheinlichkeit", legend=false);

plot_variable(model_chain, :o, STEP; label=["x" "y" "z"], xlabel="Iteration [÷ $(STEP)]", ylabel="Zugehörigkeit", legend=false)
density_variable(model_chain, :o; label=["x" "y" "z"], xlabel="Zugehörigkeit [0,1]", ylabel="Wahrscheinlichkeit", legend=false);

plot(
    plt_t_chain, plt_t_dens,
    plt_r_chain, plt_r_dens,
    layout=(2, 2)
)

# scatter_position(model_chain, 100)
# plot_logprob(model_chain, STEP)

# # pyplot()
# polar_histogram_variable(model_chain, :r; nbins=180)
# # mean(getproperty.(variables.(model_chain), (:t)))
# plot_depth_img(render(render_context, scene, parameters.object_id, to_pose(model_chain[end].variables.t, model_chain[end].variables.r)) |> Array)
