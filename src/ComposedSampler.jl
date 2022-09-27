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

The samplers are cycled systematically using the sampler indices in `order`.
You could provide a custom order, e.g. `[1, 2, 2]`, which uses sampler 1 once and then sampler 2 twice.
"""
struct ComposedSampler{O<:AbstractVector{<:Integer},T<:Tuple} <: AbstractMCMC.AbstractSampler
    order::O
    samplers::T
end

"""
    ComposedSampler(order, samplers)
Construct a ComposedSampler from an order which can be a vector / range of the sampler indices and an arbitrary number of samplers.
"""
ComposedSampler(order::AbstractVector{<:Integer}, samplers::AbstractMCMC.AbstractSampler...) = ComposedSampler(order, samplers)

"""
    ComposedSampler(order, samplers)
Construct a ComposedSampler from an arbitrary number of samplers.
The sampler order is `1:length(samplers)` by default.
"""
ComposedSampler(samplers::AbstractMCMC.AbstractSampler...) = ComposedSampler(1:length(samplers), samplers...)

"""
    ComposedSamplerState
Additionally to the current sample, the current index for cycling the samplers is stored.
"""
struct ComposedSamplerState{S<:Sample}
    sample::S
    index::Int64
end

function Base.show(io::IO, g::ComposedSampler)
    println(io, "ComposedSampler(order $(g.order), internal samplers:")
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
function AbstractMCMC.step(rng::AbstractRNG, model, sampler::ComposedSampler)
    # All samplers have the same prior
    sample, _ = AbstractMCMC.step(rng, model, first(sampler.samplers))
    # sample, state
    sample, ComposedSamplerState(sample, 1)
end

"""
    step(sample, model, sampler, state)
Implementing the AbstractMCMC interface for steps given a state from the last step.
"""
function AbstractMCMC.step(rng::AbstractRNG, model, sampler::ComposedSampler, state::ComposedSamplerState)
    index = sampler.order[state.index]
    current_sampler = sampler.samplers[index]
    # TODO Different states like multiple weighted samples not supported yet
    sample, _ = AbstractMCMC.step(rng, model, current_sampler, state.sample)
    # increment sample cycling index, reset to 1 if end is reached
    new_index = state.index == length(sampler.order) ? 1 : state.index + 1
    # sample, state
    sample, ComposedSamplerState(sample, new_index)
end
