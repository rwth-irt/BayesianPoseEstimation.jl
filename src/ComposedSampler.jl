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

If the ComposedSampler is `systematic`, the samplers are cycled in order (practical, according to Wikipedia), otherwise a random sampler is selected based on the `weights`.
Use `weights` for example to sample from the prior only every once in a while which might be inefficient but help with escaping local minima.  
"""
struct ComposedSampler{O<:AbstractVector{<:Integer},T<:Tuple} <: AbstractMCMC.AbstractSampler
    # TODO doc
    order::O
    samplers::T
end

# TODO doc
function ComposedSampler(order::AbstractVector{<:Integer}, samplers::AbstractMCMC.AbstractSampler...)
    if !allequal(prior.(samplers))
        throw(ArgumentError("ComposedSampler: different priors for samplers not supported since they would require different bijectors"))
    end
    ComposedSampler(order, samplers)
end
# TODO doc
ComposedSampler(samplers::AbstractMCMC.AbstractSampler...) = ComposedSampler(1:length(samplers), samplers...)

# TODO doc
struct ComposedSamplerState{S<:Sample}
    sample::S
    i_sampler::Int64
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
Proposes one sample from the prior distribution of the model.
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
    index = sampler.order[state.i_sampler]
    current_sampler = sampler.samplers[index]
    # TODO Different states like multiple weighted samples not supported yet
    sample, _ = AbstractMCMC.step(rng, model, current_sampler, state.sample)
    # sample, state
    i_sampler = state.i_sampler == length(sampler.order) ? 1 : state.i_sampler + 1
    sample, ComposedSamplerState(sample, i_sampler)
end
