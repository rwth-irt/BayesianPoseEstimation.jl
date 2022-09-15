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
struct MetropolisHastings{Q,P<:AbstractProposal} <: AbstractMCMC.AbstractSampler
    prior::Q
    proposal::P
end

Base.show(io::IO, mh::MetropolisHastings) = print(io, "MetropolisHastings with proposal:\n$(model(proposal(mh)))")

"""
    proposal(mh)
Get the proposal model of the Sampler.
"""
proposal(mh::MetropolisHastings) = mh.proposal

# TODO required in Gibbs?
"""
    proposal(mh)
Set the proposal model of the Sampler.
"""
set_proposal(mh::MetropolisHastings, q::AbstractProposal) = @set mh.proposal = q

"""
    step(rng, model, sampler)
Implementing the AbstractMCMC interface for the initial step.
Proposes one sample from the prior of the sampler.
"""
function AbstractMCMC.step(rng::AbstractRNG, model, mh::MetropolisHastings)
    sample = rand(rng, mh.prior)
    # TODO Why is conversion necessary?
    state = @set sample.logp = logdensityof(model, sample) |> Float64
    # sample, state are the same for MH
    state, state
end

"""
    step(sample, model, sampler, state)
Implementing the AbstractMCMC interface for steps given a state from the last step.
"""
function AbstractMCMC.step(rng::AbstractRNG, model, sampler::MetropolisHastings, state::Sample)
    # If previous was Gibbs, it has an infinite probability to be accepted -> calulate the actual logdensity
    if isinf(log_prob(state))
        state = @set state.logp = logdensityof(model, state) |> Float64
    end
    # propose new sample
    proposed = propose(rng, sampler.proposal, state)
    proposed = @set proposed.logp = logdensityof(model, proposed) |> Float64
    # acceptance ratio
    α = (log_prob(proposed) -
         log_prob(state) +
         transition_probability(proposal(sampler), state, proposed) -
         transition_probability(proposal(sampler), proposed, state))
    if log(rand(rng)) > α
        # reject
        return state, state
    else
        # accept (always the case if difference positive since log([0,1])->[-inf,0])
        return proposed, proposed
    end
end
