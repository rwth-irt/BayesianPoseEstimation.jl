# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# TODO move usings
using DensityInterface
using Random
using Rotations
using SciGL

"""
    PriorModel(render_context, scene, object_id, t_model, r_model, o_model)
Creates a RenderModel{IndependentModel} for the variables t, r & o
"""
struct PriorModel{M}
    model::M
end
PriorModel(render_context::RenderContext, scene::Scene, object_id::Integer, t_model, r_model, o_model) = PriorModel( RenderModel(render_context, scene, object_id, IndependentModel((; t=t_model, r=r_model, o=o_model))))

Base.rand(rng::AbstractRNG, prior::PriorModel, dims::Integer...) = rand(rng, prior.model, dims...)

@inline DensityKind(::PriorModel) = HasDensity()
DensityInterface.logdensityof(prior::PriorModel, x) = logdensityof(prior.model, x)

Bijectors.bijector(prior::PriorModel) = bijector(prior.model)


"""
    PosteriorModel
Consist of a `prior_model`, which generates a sample with variables t, r, o & μ.
The `observation_model` is a function of (μ, o) which creates an ObservationModel.
"""
struct PosteriorModel{P<:PriorModel,O}
    prior_model::P
    # Expected to be f(μ, o)
    observation_model::O
end

function Base.rand(rng::AbstractRNG, model::PosteriorModel, dims::Integer...)
    prior_sample = rand(rng, model.prior_model, dims...)
    observation_instance = model.observation_model(variables(prior_sample).μ, variables(prior_sample).o)
    z = rand(rng, observation_instance)
    merge(prior_sample, (; z=z))
end

# DensityInterface
@inline DensityKind(::PosteriorModel) = HasDensity()
function DensityInterface.logdensityof(model::PosteriorModel, sample)
    ℓ_prior = logdensityof(model.prior_model, sample)
    observation_instance = model.observation_model(variables(sample).μ, variables(sample).o)
    ℓ_likelihood = logdensityof(observation_instance, variables(sample).z)
    .+(promote(ℓ_prior, ℓ_likelihood)...)
end
