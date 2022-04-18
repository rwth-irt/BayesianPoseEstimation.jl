# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 
# Samples

using Random
using TransformVariables

# TODO Logjac does not seem to make sense for Priors on first sight but 

"""
    AbstractPriorModel
Implement Random.rand(rng, type, model) => Sample and MeasureTheory.logdensity(model, sample) => Real
"""
abstract type AbstractPriorModel end
Random.rand(m::AbstractPriorModel) = rand(Random.GLOBAL_RNG, m)

"""
    IndependentPrior
Model variables are independent so the sampling order does not matter
"""
struct IndependentPrior{K,V} <: AbstractPriorModel
    models::NamedTuple{K,V}
end

"""
    models(prior)
Returns the 
"""
models(prior::IndependentPrior) = prior.models

"""
    rand(rng, prior)
Create a random sample  from the IndependentPrior.
"""
function Random.rand(rng::AbstractRNG, prior::IndependentPrior{K}) where {K}
    vars = map(prior.models) do m
        SampleVariable(rng, m)
    end
    Sample(NamedTuple{K}(vars), -Inf)
end

# TODO probably move
"""
    maybe_cpu(A)
Copies the CuArray to the cpu, otherwise returns the input 
"""
maybe_cpu(A) = A
maybe_cpu(A::AbstractArray) = Array(A)

"""
    logdensity(models, sample)
Evaluate the logdensity in the model domain.
Logjac correction is automatically applied to SampleVariables but not to any other AbstractVariable
"""
function MeasureTheory.logdensity(prior::IndependentPrior, sample::Sample)
    v = vars(sample)
    ℓ = map_models(logdensity_var, prior.models, v)
    reduce(.+, maybe_cpu.(ℓ))
end

# WARN logpdf is not typestable due to MeasureTheory internals. Probably not used anyways. so not implemented
