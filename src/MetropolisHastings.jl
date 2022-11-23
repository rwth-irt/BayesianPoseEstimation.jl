# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using AbstractMCMC
using Random

"""
    MetropolisHastings
Different MetropolisHastings samplers only differ by their proposals.   
"""
struct MetropolisHastings{Q} <: AbstractMCMC.AbstractSampler
    proposal::Q
end

Base.show(io::IO, ::MetropolisHastings) = print(io, "MetropolisHastings")

"""
    proposal(mh)
Get the proposal model of the Sampler.
"""
nodes(mh::MetropolisHastings) = mh.proposal

"""
    step(rng, model, sampler)
Implementing the AbstractMCMC interface for the initial step.
Proposes one sample from the prior of the sampler.
"""
function AbstractMCMC.step(::AbstractRNG, model::PosteriorModel, sampler::MetropolisHastings)
    # rand on PosteriorModel samples from prior in unconstrained domain
    prior_sample = rand(model)
    # initial evaluation of the posterior logdensity
    ℓ_sample = Sample(variables(prior_sample), Float64(logdensityof(model, prior_sample)))
    # sample, state are the same for MH
    ℓ_sample, ℓ_sample
end

"""
    step(rng, model, sampler, state)
Implementing the AbstractMCMC interface for steps given a state from the last step.
"""
function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, sampler::MetropolisHastings, state::Sample)
    # If previous was Gibbs, it has an infinite probability to be accepted -> calulate the actual logdensity
    if isinf(logprob(state))
        state = Sample(variables(state), Float64(logdensityof(model, state)))
    end
    # propose new sample after calculating the logdensity of the previous one since the render buffer is overwritten
    proposed = propose(sampler.proposal, state)
    sample = Sample(variables(proposed), Float64(logdensityof(model, proposed)))
    # acceptance ratio
    α = (logprob(sample) -
         logprob(state) +
         transition_probability(nodes(sampler), state, sample) -
         transition_probability(nodes(sampler), sample, state))
    if log(rand(rng)) > α
        # reject
        return state, state
    else
        # accept (always the case if difference positive since log([0,1])->[-inf,0])
        return sample, sample
    end
end
