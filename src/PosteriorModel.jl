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
