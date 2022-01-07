# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using AbstractMCMC
using Accessors
using Random

#TODO I will probably want to use it for other samplers, too. Move somewhere else

"""
    MetropolisHastings
Different MetropolisHastings samplers only differ by their proposals.
"""
struct MetropolisHastings{T<:AbstractProposal} <: AbstractMCMC.AbstractSampler
    q::T
end

Base.show(io::IO, mh::MetropolisHastings) = print(io, "MetropolisHastings with proposal:\n$(model(mh.q))")

"""
    proposal(mh)
Get the proposal model of the Sampler.
"""
proposal(mh::MetropolisHastings) = mh.q

"""
    proposal(mh)
Set the proposal model of the Sampler.
"""
set_proposal(mh::MetropolisHastings, q::AbstractProposal) = @set mh.q = q

"""
    propose(rng, m, s)
Propose a new sample for the MetropolisHastings sampler.
"""
propose(rng::AbstractRNG, m::MetropolisHastings, s) = propose(rng, m.q, s)

"""
    step(rng, model, sampler)
Implementing the AbstractMCMC interface for the initial step.
Proposes one sample from the prior distribution of the PosteriorModel
"""
function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, ::MetropolisHastings)
    sample = propose(rng, IndependentProposal(model.q))
    state = @set sample.p = logdensity(model, sample)
    # sample, state are the same for MH
    state, state
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
