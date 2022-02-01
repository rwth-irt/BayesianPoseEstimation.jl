# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using AbstractMCMC
using Accessors
using Random

"""
    MetropolisHastings
Different MetropolisHastings samplers only differ by their proposals.
"""
struct MetropolisHastings{T<:AbstractProposal} <: AbstractMCMC.AbstractSampler
    initial::IndependentProposal
    proposal::T
end

Base.show(io::IO, mh::MetropolisHastings) = print(io, "MetropolisHastings with proposal:\n$(model(proposal(mh)))")

"""
    proposal(mh)
Get the proposal model of the Sampler.
"""
proposal(mh::MetropolisHastings) = mh.proposal

"""
    proposal(mh)
Set the proposal model of the Sampler.
"""
set_proposal(mh::MetropolisHastings, q::AbstractProposal) = @set mh.proposal = q

"""
    propose(rng, m, s)
Propose a sample without a previous sample for the MetropolisHastings sampler.
"""
propose(rng::AbstractRNG, mh::MetropolisHastings) = propose(rng, mh.initial)

"""
    propose(rng, m, s)
Propose a new sample for the MetropolisHastings sampler.
"""
propose(rng::AbstractRNG, mh::MetropolisHastings, s) = propose(rng, mh.proposal, s)

"""
    step(rng, model, sampler)
Implementing the AbstractMCMC interface for the initial step.
Proposes one sample from the initial proposal model of `mh`.
"""
function AbstractMCMC.step(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, mh::MetropolisHastings)
    sample = propose(rng, mh)
    # TODO wasteful? Make p mutable or generate
    state = @set sample.p = logdensity(model, sample)
    # sample, state are the same for MH
    state, state
end

"""
    step(sample, model, sampler, state)
Implementing the AbstractMCMC interface for steps given a state from the last step.
"""
function AbstractMCMC.step(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, sampler::MetropolisHastings, state::Sample)
    # old sample requires a valid log probability
    if isinf(log_probability(state))
        state = @set state.p = logdensity(model, state)
    end
    # propose new sample
    sample = propose(rng, sampler, state)
    sample = @set sample.p = logdensity(model, sample)
    # acceptance ratio
    α = (log_probability(sample) -
         log_probability(state) +
         transition_probability(proposal(sampler), state, sample) -
         transition_probability(proposal(sampler), sample, state))
    if log(rand(rng)) > α
        # reject
        return state, state
    else
        # accept (always the case if difference positive since log([0,1])->[-inf,0])
        return sample, sample
    end
end
