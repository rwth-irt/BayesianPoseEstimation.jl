# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using AbstractMCMC
using Random

"""
    MetropolisHastings
Different MetropolisHastings samplers only differ by their proposals.

    MetropolisHastings(prior, proposal::AbstractProposal)
Generally, proposals require the prior to be transformed to ℝⁿ and account for the logjac correction.

    MetropolisHastings(prior, proposal::IndependentProposal)
The proposal must be in the model domain of the prior and is typically the prior itself.
    
"""
struct MetropolisHastings{Q,B<:NamedTuple{<:Any,<:Tuple{Vararg{<:Bijector}}},P<:AbstractProposal} <: AbstractMCMC.AbstractSampler
    prior::Q
    bijectors::B
    proposal::P
end

function MetropolisHastings(prior, proposal::AbstractProposal)
    if is_constrained(proposal)
        MetropolisHastings(prior, (;), proposal)
    else
        # Bijectors if unconstrained
        MetropolisHastings(prior, map_materialize(bijector(prior)), proposal)
    end
end

"""
    map_materialize(bijectors)
Maps Broadcast.materialize over a collection of bijectors.
Falls back to materialize without map for non-collections.
"""
map_materialize(bijector) = Broadcast.materialize(bijector)
map_materialize(bijectors::Union{NamedTuple,Tuple,AbstractArray}) = map(Broadcast.materialize, bijectors)

Base.show(io::IO, mh::MetropolisHastings) = print(io, "MetropolisHastings with bijectors for $(keys(mh.bijectors))")

"""
    proposal(mh)
Get the proposal model of the Sampler.
"""
proposal(mh::MetropolisHastings) = mh.proposal

"""
    logprob(model, mh, sample)
Takes care of transforming the sample according to the bijector of mh and adding the logjac correction.
"""
function logprob(model, mh::MetropolisHastings, sample::Sample)
    model_sample, logjac = to_model_domain(sample, mh.bijectors)
    logdensityof(model, model_sample) + logjac
end

"""
    step(rng, model, sampler)
Implementing the AbstractMCMC interface for the initial step.
Proposes one sample from the prior of the sampler.
"""
function AbstractMCMC.step(rng::AbstractRNG, model, sampler::MetropolisHastings)
    prior_sample = rand(rng, sampler.prior)
    unconstrained_sample = to_unconstrained_domain(prior_sample, sampler.bijectors)
    sample = Sample(variables(unconstrained_sample), Float64(logprob(model, sampler, unconstrained_sample)))
    # sample, state are the same for MH
    sample, sample
end

"""
    step(sample, model, sampler, state)
Implementing the AbstractMCMC interface for steps given a state from the last step.
"""
function AbstractMCMC.step(rng::AbstractRNG, model, sampler::MetropolisHastings, state::Sample)
    # If previous was Gibbs, it has an infinite probability to be accepted -> calulate the actual logdensity
    if isinf(logprob(state))
        state = Sample(variables(state), Float64(logprob(model, sampler, state)))
    end
    # propose new sample after calculating the logdensity of the previous one since the render buffer is overwritten
    sample = propose(rng, sampler.proposal, state)
    sample = Sample(variables(sample), Float64(logprob(model, sampler, sample)))
    # acceptance ratio
    α = (logprob(sample) -
         logprob(state) +
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
