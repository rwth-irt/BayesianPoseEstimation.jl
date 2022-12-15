# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using AbstractMCMC
using DensityInterface
using Random

"""
    Gibbs
Analytically samples groups of variables conditioned on the other variables.
Thus, the samples are always accepted.
"""
struct Gibbs{P,E,B} <: AbstractMCMC.AbstractSampler
    # Analytic proposal
    proposal::P
    evaluation::E
    bijectors::B
end
Gibbs(proposal_model, posterior_model) = proposal(Gibbs, proposal_model, posterior_model)

Base.show(io::IO, gibbs::Gibbs) = print(io, "Gibbs(bijectors for $(keys(gibbs.bijectors)), model $(gibbs.proposal))")

# Gibbs can act as proposal

# The variables of the Gibbs proposal would be skipped if they existed in the previous sample
remove_variables(::Gibbs{<:AbstractNode{name}}, variables::NamedTuple) where {name} = Base.structdiff(variables, (; name => ()))
remove_variables(gibbs::Gibbs{<:SequentializedGraph}, variables::NamedTuple) = Base.structdiff(variables, gibbs.proposal)


"""
    propose(proposal, sample)
Independent samples are just random values from the model.
"""
function propose(gibbs::Gibbs, sample::Sample)
    # Proposal conditioned on the sample in the model domain
    model_sample, _ = to_model_domain(sample, gibbs.bijectors)
    model_variables = remove_variables(gibbs, variables(model_sample))
    proposed = rand(gibbs.proposal, model_variables)
    evaluated = evaluate(gibbs.evaluation, proposed)
    # Sampling in unconstrained domain
    to_unconstrained_domain(Sample(evaluated), gibbs.bijectors)
end

"""
    transition_probability(proposal, new_sample, prev_sample)
For Gibbs proposals, all samples will be accepted.
"""
transition_probability(::Gibbs, ::Sample, ::Sample) = Inf

# AbstractModel

"""
  step(rng, model, sampler)
Implementing the AbstractMCMC interface for the initial step.
Proposes one sample from the prior distribution of the model.
"""
function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, sampler::Gibbs)
    # rand on PosteriorModel samples from prior in unconstrained domain
    s = rand(model)
    # initial evaluation of the posterior logdensity
    s = set_logp(s, logdensityof(model, s))
    # sample, state are the same for Gibbs
    s, s
end

"""
    step(sample, model, sampler, state)
Implementing the AbstractMCMC interface for steps given a state from the last step.
AnalyticalGibbs always accepts the sample, since it is always the best possible sample given the prior sample
"""
function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, sampler::Gibbs, state::Sample)
    proposed = propose(sampler, state)
    # Even though it is always accepted different samplers expect a valid log probability for the previous sample to avoid re-evaluating the logdensity multiple times
    proposed = set_logp(proposed, logdensityof(model, proposed))
    # sample, state
    proposed, proposed
end
