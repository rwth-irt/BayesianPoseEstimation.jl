# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 
# Samples

# TODO Do I want another wrapper or base all these models on AbstractModel?
using AbstractMCMC: AbstractModel
using Accessors
using Bijectors
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

Bijectors.bijector(model::IndependentModel) = map(bijector, models(model))
Bijectors.transformed(model::IndependentModel) = IndependentModel(map(transformed, models(model)))

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

Bijectors.bijector(model::RngModel) = bijector(model.model)
Bijectors.transformed(model::RngModel) = @set model.model = transformed(model.model)

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

Bijectors.bijector(model::ComposedModel) = merge(bijector.(model.models)...)
Bijectors.transformed(model::ComposedModel) = @set model.models = transformed.(model.models)

"""
    ConditionedModel
Decorator for a model which conditiones the sample on the data before evaluating the logdensity.
"""
struct ConditionedModel{T,V,M} <: AbstractModel
    data::NamedTuple{T,V}
    model::M
end

ConditionedModel(sample::Sample, model) = ConditionedModel(sample.variables, model)

function Base.rand(rng::AbstractRNG, model::ConditionedModel, dims...)
    sample = rand(rng, model.model, dims...)
    merge(sample, model.data)
end

@inline DensityKind(::ConditionedModel) = HasDensity()

"""
    logdensityof(model, sample)
Calculates logdensity of each inner model and sums up these individual logdensities.
"""
function DensityInterface.logdensityof(model::ConditionedModel, sample)
    conditioned_sample = merge(sample, model.data)
    logdensityof(model.model, conditioned_sample)
end

Bijectors.bijector(model::ConditionedModel) = Base.structdiff(bijector(model.model), model.data)
Bijectors.transformed(model::ConditionedModel) = TransformedConditionedModel(model.data, model.model)

# TODO workaround - is there a cleaner possibilty without making any assumptions about the model? e.g. that each variable is only defined once, model is one of these model or a distribution...?

struct TransformedConditionedModel{T,V,M} <: AbstractModel
    data::NamedTuple{T,V}
    model::M
end

TransformedConditionedModel(sample::Sample, model) = TransformedConditionedModel(sample.variables, model)

function Base.rand(rng::AbstractRNG, model::TransformedConditionedModel, dims...)
    sample = rand(rng, model.model, dims...)
    merged = merge(sample, model.data)
    # Transform everything except data
    to_unconstrained_domain(merged, bijector(model))
end

@inline DensityKind(::TransformedConditionedModel) = HasDensity()

function DensityInterface.logdensityof(model::TransformedConditionedModel, sample)
    # Transform sample back to model domain
    transformed_sample, logjac = to_model_domain(sample, bijector(model))
    conditioned_sample = merge(transformed_sample, model.data)
    logdensityof(model.model, conditioned_sample) + logjac
end

Bijectors.bijector(model::TransformedConditionedModel) = Base.structdiff(bijector(model.model), model.data)
Bijectors.transformed(model::TransformedConditionedModel) = ConditionedModel(model.data, model.model) 
