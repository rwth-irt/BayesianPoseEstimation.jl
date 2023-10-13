# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    CoordinateSampler
Provide a different sampler for each group (just like in Turing.jl).
This way each variable can be sampled with the most appropriate sampler (e.g. using gradients or analytically calculating the conditional posterior).

It loops over all samplers in an inference step, e.g. required in filter applications.
"""
struct CoordinateSampler{T<:Tuple} <: AbstractMCMC.AbstractSampler
    samplers::T
end
CoordinateSampler(s1, samplers...) = CoordinateSampler((s1, samplers...))

function Base.show(io::IO, g::CoordinateSampler)
    println(io, "CoordinateSampler(internal samplers:")
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
function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, sampler::CoordinateSampler)
    # All samplers use the prior of the PosteriorModel
    new_sample, new_state = AbstractMCMC.step(rng, model, first(sampler.samplers))
    # sample, state
    new_sample, new_state
end

"""
    step(sample, model, sampler, state)
Implementing the AbstractMCMC interface for steps given a state from the last step.
"""
function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, sampler::CoordinateSampler, state)
    for s in sampler.samplers
        sample, state = AbstractMCMC.step(rng, model, s, state)
    end
    sample, state
end
