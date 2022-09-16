# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using Accessors
using CUDA
using Distributions
using LinearAlgebra
using MCMCDepth
using Plots
using Random
using SciGL
using Test

parameters = Parameters()
parameters = @set parameters.mesh_files = ["meshes/BM067R.obj"]
parameters = @set parameters.rotation_type = :RotXYZ
parameters = @set parameters.device = :CUDA

const PLOT = true
if PLOT
    pyplot()
end
maybe_plot(fn, x...; y...) = PLOT ? fn(x...; y...) : nothing
rng = Random.default_rng()
Random.seed!(rng, 42)
dev_rng = device_rng(parameters)
Random.seed!(dev_rng, 42)
CUDA.allowscalar(false)

# Setup render context & scene
render_context = RenderContext(parameters.width, parameters.height, parameters.depth, device_array_type(parameters))
# CvCamera like ROS looks down positive z
scene = Scene(parameters, render_context)
t = [-0.05, 0.05, 0.25]
r = normalize([1, 1, 0])
p = to_pose(t, r, parameters.rotation_type)
μ = @inferred render(render_context, scene, 1, p)
maybe_plot(plot_depth_img, Array(μ))
@test size(μ) == (100, 100)
@test eltype(μ) == Float32
@test maximum(μ) > 0

# Proposal model
t_model = RngModel(rng, ProductBroadcastedDistribution(KernelNormal, zeros(parameters.precision, 3), parameters.precision.(parameters.proposal_σ_t)))
r_model = RngModel(rng, ProductBroadcastedDistribution(KernelNormal, zeros(parameters.precision, 3), parameters.precision.(parameters.proposal_σ_r)))

prior = IndependentModel((; t=t_model, r=r_model))
prior_sample = @inferred rand(dev_rng, prior)
@test variables(prior_sample).t isa Vector{parameters.precision}
@test variables(prior_sample).r isa Vector{parameters.precision}

tr_proposal = IndependentProposal(prior)
Random.seed!(rng, 42)
proposed_sample = @inferred propose(dev_rng, tr_proposal, prior_sample)
@test variables(proposed_sample).r isa Vector{parameters.precision}
@test variables(proposed_sample).t isa Vector{parameters.precision}
ℓ_proposed = @inferred transition_probability(tr_proposal, prior_sample, proposed_sample)

# RenderProposal
render_proposal = @inferred RenderProposal(parameters.rotation_type, render_context, scene, parameters.object_id, tr_proposal)
Random.seed!(rng, 42)
Random.seed!(dev_rng, 42)
render_sample = @inferred propose(dev_rng, render_proposal, prior_sample)
ℓ_rendered = @inferred transition_probability(tr_proposal, prior_sample, render_sample)

@test variables(proposed_sample).t == variables(render_sample).t
@test variables(proposed_sample).r == variables(render_sample).r
@test variables(render_sample).μ isa array_for_rng(dev_rng){parameters.precision,2}
@test ℓ_proposed == ℓ_rendered

# Random interface to sample from models which are not proposals (independent or conditioned)
render_sample = @inferred rand(dev_rng, render_proposal)
ℓ_rendered = @inferred transition_probability(tr_proposal, prior_sample, render_sample)

@test variables(proposed_sample).t != variables(render_sample).t
@test variables(proposed_sample).r != variables(render_sample).r
@test variables(render_sample).μ isa array_for_rng(dev_rng){parameters.precision,2}
@test ℓ_proposed == ℓ_rendered

# logdensity is always zero for symmetric, so also test IndependentProposal
independent_tr_proposal = IndependentProposal(prior)
Random.seed!(rng, 42)
Random.seed!(dev_rng, 42)
independent_proposed_sample = @inferred propose(dev_rng, independent_tr_proposal, prior_sample)
ℓ_independent_proposed = @inferred transition_probability(independent_tr_proposal, prior_sample, independent_proposed_sample)

independent_render_proposal = @inferred RenderProposal(parameters.rotation_type, render_context, scene, parameters.object_id, independent_tr_proposal)
Random.seed!(rng, 42)
Random.seed!(dev_rng, 42)
independent_rendered_sample = @inferred propose(dev_rng, independent_render_proposal, prior_sample)
ℓ_independent_rendered = @inferred transition_probability(independent_render_proposal, prior_sample, independent_rendered_sample)

@test variables(independent_proposed_sample).t == variables(independent_rendered_sample).t
@test variables(independent_proposed_sample).r == variables(independent_rendered_sample).r
@test variables(independent_rendered_sample).μ isa array_for_rng(dev_rng){parameters.precision,2}
@test ℓ_independent_proposed == ℓ_independent_rendered
