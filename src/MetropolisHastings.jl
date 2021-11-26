# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using AbstractMCMC
using Accessors
using MeasureTheory
using Random

#TODO I will probably want to use it for other samplers, too. Move somewhere else
"""
    PosteriorModel
Models the posterior logdensity p(θ|y)~ℓ(y|θ)q(θ) up to a constant.
`q` is the prior model and should support a rand(q) and logdensity(q, θ).
`ℓ` is the observation model / likelihood for a sample.
"""
struct PosteriorModel <: AbstractMCMC.AbstractModel
    # TODO constrain types?
    q
    ℓ
end

"""
    logdensity(m, s)
Non-corrected logdensity of the of the sample `s` given the measure `m`.
"""
MeasureTheory.logdensity(m::PosteriorModel, s::Sample) =
    logdensity(m.q, s) + logdensity(m.ℓ, s)

"""
    MetropolisHastings
Different MetropolisHastings samplers only differ by their proposals.
"""
struct MetropolisHastings{T<:AbstractProposal} <: AbstractMCMC.AbstractSampler
    q::T
end

"""
    propose(rng, m, s)
Propose a new sample for the MetropolisHastings sampler.
"""
propose(rng::AbstractRNG, m::MetropolisHastings, s) = propose(rng, m.q, s)

"""
    propose(m, s)
Propose a new sample for the MetropolisHastings sampler.
"""
propose(m::MetropolisHastings, s) = propose(Random.GLOBAL_RNG, m.q, s)

"""
    step(rng, model, sampler)
Implementing the AbstractMCMC interface for the initial step.
Proposes one sample from the prior distribution of the PosteriorModel
"""
function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, ::MetropolisHastings)
    sample = propose(rng, IndependentProposal(model.q))
    state = @set sample.p = logdensity(model, sample)
    # sample, state are the same for MH
    return state, state
end

"""
    step(sample, model, sampler, state)
Implementing the AbstractMCMC interface for steps given a state from the last step.
"""
function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, sampler::MetropolisHastings, state::Sample)
    # propose new sample
    sample = propose(rng, sampler, state)
    proposal = @set sample.p = logdensity(model, sample)
    # acceptance ratio
    α = (log_probability(proposal) -
         log_probability(state) +
         transition_probability(sampler.q, state, proposal) -
         transition_probability(sampler.q, proposal, state))
    if log(rand(rng)) > α
        # reject
        return state, state
    else
        # accept (always the case if difference positive since log([0,1])->[-inf,0])
        return proposal, proposal
    end
end
