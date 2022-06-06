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
- DensityInterface.logdensityof(model, sample::Sample)::Real.
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

function Base.rand(rng::AbstractRNG, model::IndependentModel, dims::Integer...)
    var_nt = map(model.models) do m
        # ModelVariables are more natural / efficient to sample. It should be up to the sampler logic to decide whether the transformation overhead is required. 
        ModelVariable(rng, m, dims...)
    end
    Sample(NamedTuple(var_nt), -Inf)
end

"""
    logdensityof(model, sample)
Maps `logdensityof` over models and variables with matching names.
Uses 0.0 as if the variable name is non-existent in the sample.
Note that all variables are assumed to be independent and vectorization is accounted for by broadcasting.
"""
DensityInterface.logdensityof(model::IndependentModel{T}, sample::Sample) where {T} = reduce(.+, (map_intersect_(logdensityof, model.models, vars(sample), 0.0)))
    # TODO Really don't know, why this would be type-instable
    # map_intersect(logdensityof, model.models, vars(sample), 0.0)
    # TODO mapreduce is also type instable?


# TODO What about Gibbs → Sampling conditioned subset? In my case, the proposed variables are independent, so it does not make sense to implement it  at the moment. I would probably implement a new GibbsModel to avoid rand(::KernelDistribution) having to return a Sample.
