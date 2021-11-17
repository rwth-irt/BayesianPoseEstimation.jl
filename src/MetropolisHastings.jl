# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using AbstractMCMC
using Accessors
using Random

#TODO I will probably want to use it for other samplers, too. Move somewhere else
"""
    PosteriorModel
Has a data conditioned function to evaluate the posterior density up to a constant ℓ(θ)~p(y|θ)p(θ) 
"""
struct PosteriorModel{T<:Function} <: AbstractMCMC.AbstractModel
    ℓ::T
end

"""
    MetropolisHastings
Different MetropolisHastings samplers only differ by their proposals.
"""
struct MetropolisHastings{T<:AbstractProposal} <: AbstractMCMC.AbstractSampler
    q::T
end

"""
    step(sample, log_density, sampler, state)
Implementing the AbstractMCMC interface.
"""
function step(rng::AbstractRNG, model::PosteriorModel, sampler::MetropolisHastings, state::AbstractSample)
    # propose new sample
    s = propose(sampler.q, state)
    proposal = @set s.ℓ = model.ℓ
    # acceptance ratio
    α = (proposal.ℓ -
         state.ℓ +
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
