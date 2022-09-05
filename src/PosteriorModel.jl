# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# TODO move usings
using DensityInterface
using Random
using Rotations
using SciGL

# TODO doc

# TODO Keep it here?
PriorModel(t_model, r_model, o_model) = IndependentModel((t=t_model, r=r_model, o=o_model))

struct PosteriorModel{R<:Rotation,P,O,C<:RenderContext,S<:Scene}
    prior_model::P
    # Expected to be f(μ, o)
    observation_model::O
    # Render related objects
    render_context::C
    scene::S
    object_id::Int
end

PosteriorModel(prior_model::P, observation_model::O, render_context::C, scene::S, object_id::Int, ::Type{R}) where {P,O,C<:RenderContext,S<:Scene,R<:Rotation} = PosteriorModel{R,P,O,C,S}(prior_model, observation_model, render_context, scene, object_id)

function Base.rand(rng::AbstractRNG, model::PosteriorModel{R}, dims::Integer...) where {R}
    # WARN should not be used during inference, instead rand from the prior.
    prior_sample = rand(rng, model.prior_model, dims...)
    render_sample = render(model.render_context, model.scene, model.object_id, R, prior_sample)
    observation_instance = model.observation_model(variables(render_sample).μ, variables(render_sample).o)
    z = rand(rng, observation_instance)
    merge(render_sample, (; z=z))
end

function render(render_context::RenderContext, scene::Scene, object_id::Integer, rotation_type::Type, sample::Sample)
    p = to_pose(variables(sample).t, variables(sample).r, rotation_type)
    μ = render(render_context, scene, object_id, p)
    # μ is only a view of the render_data. Storing it in every sample is cheap.
    merge(sample, (; μ=μ))
end

# DensityInterface
@inline DensityKind(::PosteriorModel) = HasDensity()
function DensityInterface.logdensityof(model::PosteriorModel, sample)
    ℓ_prior = logdensityof(model.prior_model, sample)
    observation_instance = model.observation_model(variables(sample).μ, variables(sample).o)
    ℓ_likelihood = logdensityof(observation_instance, variables(sample).z)
    .+(promote(ℓ_prior, ℓ_likelihood)...)
end

# TODO move to it's own file?
# TODO Parametric
# TODO Test
struct RenderProposal#{T<:Rotation}
    # Render related objects
    render_context
    scene
    object_id
    rotation_type#::T
    # Probabilistic model
    proposal
end

# TODO hard-coded variable names? t, o, μ
function propose(rng::AbstractRNG, proposal::RenderProposal, dims...)
    sample = propose(rng, proposal.proposal, dims...)
    p = to_pose(sample.t, sample.r, proposal.rotation_type)
    # μ is only a view of the render_data. Storing it in every sample is cheap.
    μ = render(proposal.render_context, proposal.scene, object_id, p)
    merge(sample, (; μ=μ))
end

transition_probability(proposal::RenderProposal, new_sample, prev_sample) = transition_probability(proposal.proposal, new_sample, prev_sample)

# function Random.rand(rng::AbstractRNG, T::Type, prior::PoseDepthPrior)
#     tro_sample = rand(rng, T, prior.tro_model)
#     t = model_value(tro_sample, :t)
#     r = model_value(tro_sample, :o)
#     # TODO multiple hypotheses by implementing rand([rng=GLOBAL_RNG], [S], [dims...]) ? Probably best of both worlds: Render number on basis of vectorized measures in the tiled texture. For uniform interface it is probably best to include tiles in all Depth models. DepthModels.jl with AbstractDepthModel? tiles(::AbstractDepthModel)
#     μ = prior.render_fn(t, r)
#     μ_var = ModelVariable(μ, asℝ)
#     μ_sample = Sample((; μ=μ_var), -Inf)
#     merge(tro_sample, μ_sample)
# end
