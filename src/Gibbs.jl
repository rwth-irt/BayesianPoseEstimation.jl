# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using AbstractMCMC
using Accessors
using MeasureTheory
using TransformVariables

struct GibbsInternal
  SamplerType
end

"""
    Gibbs
Samples groups of variables conditioned on the other variables.
Implementing this by using a different sampler for each group (just like in Turing.jl).
This way each variable can be sampled with the most appropriate sampler (e.g. using gradients or analytically calculating the conditional posterior).
"""
struct Gibbs{T<:Tuple} <: AbstractMCMC.AbstractSampler
  # First sample has to be generated independently
  initial::IndependentProposal
  # Tuple of several samplers
  samplers::T

  # Make sure that every sampler is wrapped by a GibbsSampler
  function Gibbs(initial, samplers)
    gibbsified = map(samplers) do sampler
      # avoid unnecessary double wrapping
      if isa(proposal(sampler), GibbsProposal)
        return sampler
      else
        gibbs_proposal = proposal(sampler) |> GibbsProposal
        set_proposal(sampler, gibbs_proposal)
      end
    end
    new{typeof(gibbsified)}(initial, gibbsified)
  end
end

"""
    Gibbs(sampler...)
Convenience constructor for varargs instead of Tuple of samplers
"""
Gibbs(initial::IndependentProposal, sampler::AbstractMCMC.AbstractSampler...) = Gibbs(initial, sampler)

function Base.show(io::IO, g::Gibbs)
  println(io, "Gibbs with internal samplers:")
  for sampler in g.samplers
    show(io, sampler)
  end
end

"""
    step(rng, model, sampler)
Implementing the AbstractMCMC interface for the initial step.
Proposes one sample from the prior distribution of the model.
"""
function AbstractMCMC.step(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, sampler::Gibbs)
  sample = propose(rng, sampler.initial)
  state = @set sample.logp = transform_logdensity(model, sample)
  # TODO if (ever) using something else than MH & AnalyticalGibbs, each sampler might require it's own state.
  # sample, state
  state, state
end

"""
    step(sample, model, sampler, state)
Implementing the AbstractMCMC interface for steps given a state from the last step.
Cycles through the internal samplers using a random permutation
"""
function AbstractMCMC.step(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, sampler::Gibbs, state::Sample)
  # TODO Wikipedia: practical implementations just cycle
  perm = randperm(length(sampler.samplers))
  for i in perm
    # Internal samplers decide whether to accept or reject the sample
    # Will be a recursion for internal Gibbs samplers
    _, state = AbstractMCMC.step(rng, model, sampler.samplers[i], state)
  end
  # sample, state
  state, state
end


"""
    AnalyticGibbs
Samples a group of variables analytically from a conditional posterior distribution.
Thus, a prior sample is required to condition the model on.
"""
struct AnalyticGibbs{T<:GibbsProposal{AnalyticProposal}} <: AbstractMCMC.AbstractSampler
  # First sample has to be generated independently
  initial::IndependentProposal
  # Analytic proposal
  f::T
end

"""
    proposal(ag)
Get the proposal model of the Sampler.
"""
proposal(ag::AnalyticGibbs) = ag.f

"""
    proposal(ag)
Set the proposal model of the Sampler.
"""
set_proposal(ag::AnalyticGibbs, f::GibbsProposal{AnalyticProposal}) = @set ag.f = f

"""
  propose(ag, s)
Propose an initial sample for the AnalyticGibbs sampler.
"""
propose(rng::AbstractRNG, ag::AnalyticGibbs) = propose(rng, ag.initial)

"""
  propose(ag, s)
Propose a new sample for the AnalyticGibbs sampler.
"""
propose(rng::AbstractRNG, ag::AnalyticGibbs, s) = propose(rng, ag.f, s)

"""
  step(rng, model, sampler)
Implementing the AbstractMCMC interface for the initial step.
Proposes one sample from the prior distribution of the model.
"""
function AbstractMCMC.step(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, sampler::AnalyticGibbs)
  sample = propose(rng, sampler.initial)
  state = @set sample.logp = transform_logdensity(model, sample)
  # sample, state
  state, state
end

"""
    step(sample, model, sampler, state)
Implementing the AbstractMCMC interface for steps given a state from the last step.
AnalyticalGibbs always accepts the sample, since it is always the best possible sample given the prior sample
"""
function AbstractMCMC.step(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, sampler::AnalyticGibbs, state::Sample)
  sample = propose(rng, sampler, state)
  # Even though it is always accepted different samplers expect a valid log probability for the previous sample to avoid re-evaluating the logdensity multiple times
  sample = @set sample.logp = transform_logdensity(model, sample)
  # sample, state
  sample, sample
end
