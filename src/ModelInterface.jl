# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 
# Samples

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
struct IndependentModel{var_names,T}
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

"""
    logdensityof(model, sample)
Maps `logdensityof` over models and variables with matching names.
Uses 0.0 as if the variable name is non-existent in the sample.
Note that all variables are assumed to be independent and vectorization is accounted for by broadcasting.
"""
DensityInterface.logdensityof(model::IndependentModel, sample) = .+(values(map_intersect(logdensityof, model.models, variables(sample)))...)

"""
    RngModel
Wraps an internal `model` and allows to provide an individual RNG for this model.
"""
struct RngModel{T,U<:AbstractRNG}
    rng::U
    model::T
end

"""
    rand([rng=model.rng], model, dims...)
Generate a random sample from the internal model using the rng of the model.
Ignores the rng argument.
"""
Base.rand(::AbstractRNG, model::RngModel, dims::Integer...) = rand(model.rng, model.model, dims...)
Base.rand(model::RngModel, dims::Integer...) = rand(model.rng, model, dims...)

DensityInterface.logdensityof(model::RngModel, sample) = logdensityof(model.model, sample)

"""
    ComposedModel
Generate Samples from several models which in turn generate samples by merging them in order.
"""
struct ComposedModel{T<:Tuple}
    models::T
end

ComposedModel(models...) = ComposedModel(models)

"""
    rand(rng, model, dims)
Generate a sample by sampling the internal models and merging the samples iteratively.
This means that the rightmost / last model variables are kept.
However, duplicates must be avoided, because they cannot be mapped to a unique model when evaluating logdensityof.
"""
Base.rand(rng::AbstractRNG, model::ComposedModel, dims...) = mapreduce(m -> rand(rng, m, dims...), merge, model.models)


DensityInterface.logdensityof(model::ComposedModel, sample) = mapreduce(m -> logdensityof(m, sample), +, model.models)

# TODO If I would require Gibbs sampling where the variables are not independent I would probably implement a new GibbsModel.
