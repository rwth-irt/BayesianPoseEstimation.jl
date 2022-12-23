# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using AbstractMCMC
using Accessors
using DensityInterface
using Logging
using StatsBase

"""
    ComposedSampler
Provide a different sampler for each group (just like in Turing.jl).
This way each variable can be sampled with the most appropriate sampler (e.g. using gradients or analytically calculating the conditional posterior).

The samplers are selected randomly according to the `weights` vector.
"""
struct ComposedSampler{W<:AbstractWeights,T<:Tuple} <: AbstractMCMC.AbstractSampler
    weights::W
    samplers::T
end

"""
    ComposedSampler(weights, samplers...)
Construct a ComposedSampler from a weight vector a matching number of samplers.
"""
ComposedSampler(weights::AbstractWeights, samplers::AbstractMCMC.AbstractSampler...) = ComposedSampler(weights, samplers)

"""
    ComposedSampler(samplers...)
Construct a ComposedSampler from an arbitrary number of samplers.
The weights are equal by default.
"""
ComposedSampler(samplers::AbstractMCMC.AbstractSampler...) = ComposedSampler(pweights(fill(1, length(samplers))), samplers...)

function Base.show(io::IO, g::ComposedSampler)
    println(io, "ComposedSampler(weights $(g.weights.values), internal samplers:")
    for sampler in g.samplers
        println(io, "  $(sampler)")
    end
    print(io, "")
end

"""
    step(rng, model, sampler)
Implementing the AbstractMCMC interface for the initial step.
Uses the first sampler to propose the initial sample.
"""
function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, sampler::ComposedSampler)
    # All samplers use the prior of the PosteriorModel
    new_sample, new_state = AbstractMCMC.step(rng, model, first(sampler.samplers))
    # sample, state
    new_sample, new_state
end

"""
    step(sample, model, sampler, state)
Implementing the AbstractMCMC interface for steps given a state from the last step.
"""
function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, sampler::ComposedSampler, state)
    index = sample(rng, sampler.weights)
    new_sample, new_state = AbstractMCMC.step(rng, model, sampler.samplers[index], state)
    # sample, state
    new_sample, new_state
end
