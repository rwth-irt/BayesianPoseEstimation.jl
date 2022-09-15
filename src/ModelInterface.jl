# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 
# Samples

# TODO Do I want another wrapper or base all these models on AbstractModel?
using AbstractMCMC: AbstractModel
using DensityInterface
using Random

"""
    ModelInterface
Models are the primitives to draw and evaluate samples.
A model could be as simple as containing only one variable.
- Random.rand(rng, model, dims...)::Sample
# TODO maybe rand! for optimization
- DensityInterface.logdensityof(model, sample::Sample)::Real (logjac corrected)
"""

"""
    IndependentModel{var_names, T}
Model variables are assumed to be independent and sampled in order via `ModelVariable(rng, model, dims...)`.
Does not support hierarchical models because of the one-to-one matching of the model and the variables.
"""
struct IndependentModel{var_names,T} <: AbstractModel
    # Wrapping NamedTuple to avoid type piracy
    models::NamedTuple{var_names,T}
end

"""
    models(model)
Returns the internal models of the `IndependentModel`.
"""
models(model::IndependentModel) = model.models

"""
    rand(rng, model, [dims...])
Create a new random sample from the `IndependentModel` of size `dims`.
"""
function Base.rand(rng::AbstractRNG, model::IndependentModel, dims::Integer...)
    var_nt = map(model.models) do m
        # Sampler can transform the model to propose on ℝ
        rand(rng, m, dims...)
    end
    Sample(var_nt, -Inf)
end

@inline DensityKind(::IndependentModel) = HasDensity()

"""
    logdensityof(model, sample)
Maps `logdensityof` over models and variables with matching names.
Uses 0.0 as if the variable name is non-existent in the sample.
Note that all variables are assumed to be independent and vectorization is accounted for by broadcasting.
"""
DensityInterface.logdensityof(model::IndependentModel, sample) = .+(promote(values(map_intersect(logdensityof, model.models, variables(sample)))...)...)

"""
    RngModel
Wraps an internal `model` and allows to provide an individual RNG for this model.
"""
struct RngModel{T,U<:AbstractRNG} <: AbstractModel
    rng::U
    model::T
end

model(model::RngModel) = model.model

"""
    rand([rng=model.rng], model, dims...)
Generate a random sample from the internal model using the rng of the model.
Ignores the rng argument.
"""
Base.rand(::AbstractRNG, model::RngModel, dims::Integer...) = rand(model.rng, model.model, dims...)
Base.rand(model::RngModel, dims::Integer...) = rand(model.rng, model, dims...)

@inline DensityKind(::RngModel) = HasDensity()
DensityInterface.logdensityof(model::RngModel, sample) = logdensityof(model.model, sample)

"""
    ComposedModel
Generate Samples from several models which in turn generate samples by merging them in order.
"""
struct ComposedModel{T<:Tuple} <: AbstractModel
    models::T
end

ComposedModel(models...) = ComposedModel(models)

models(model::ComposedModel) = model.models

"""
    rand(rng, model, dims)
Generate a sample by sampling the internal models and merging the samples iteratively.
This means that the rightmost / last model variables are kept.
However, duplicates must be avoided, because they cannot be mapped to a unique model when evaluating logdensityof.
"""
Base.rand(rng::AbstractRNG, model::ComposedModel, dims...) = mapreduce(m -> rand(rng, m, dims...), merge, model.models)

@inline DensityKind(::ComposedModel) = HasDensity()

"""
    logdensityof(model, sample)
Calculates logdensity of each inner model and sums up these individual logdensities.
"""
DensityInterface.logdensityof(model::ComposedModel, sample) = mapreduce(m -> logdensityof(m, sample), +, model.models)

"""
    ConditionedModel
Decorator for a model which conditiones the sample on the data before evaluating the logdensity.
"""
struct ConditionedModel{T,V,M} <: AbstractModel
    data::NamedTuple{T,V}
    model::M
end

ConditionedModel(sample::Sample, model) = ConditionedModel(sample.variables, model)

Base.rand(rng::AbstractRNG, model::ConditionedModel, dims...) = rand(rng, model.model, dims...)

@inline DensityKind(::ConditionedModel) = HasDensity()

"""
    logdensityof(model, sample)
Calculates logdensity of each inner model and sums up these individual logdensities.
"""
function DensityInterface.logdensityof(model::ConditionedModel, sample)
    conditioned_sample = merge(sample, model.data)
    logdensityof(model.model, conditioned_sample)
end
