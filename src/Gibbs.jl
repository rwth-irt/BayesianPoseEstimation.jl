# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using AbstractMCMC
using Accessors
using MeasureTheory

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
    # Tuple of several samplers
    samplers::T

    # Make sure that every sampler is wrapped by a GibbsSampler
    function Gibbs(samplers)
        gibbsified = map(samplers) do sampler
            # avoid unnecessary double wrapping
            if isa(proposal(sampler), GibbsProposal)
                return sampler
            else
                gibbs_proposal = proposal(sampler) |> GibbsProposal
                set_proposal(sampler, gibbs_proposal)
            end
        end
        new{typeof(gibbsified)}(gibbsified)
    end
end

"""
    Gibbs(sampler...)
Convenience constructor for varargs instead of Tuple of samplers
"""
Gibbs(sampler::AbstractMCMC.AbstractSampler...) = Gibbs(sampler)

# TODO remove
# """
#     gibbsify_sampler
# Wraps the proposal model in a GibbsProposal
# """
# function gibbsify_sampler(sampler::AbstractMCMC.AbstractSampler)

# end

function Base.show(io::IO, g::Gibbs)
    println(io, "Gibbs with internal samplers:")
    for sampler in g.samplers
        show(io, sampler)
    end
end

"""
    step(rng, model, sampler)
Implementing the AbstractMCMC interface for the initial step.
Proposes one sample from the prior distribution of the PosteriorModel
"""
function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, ::Gibbs)
    sample = propose(rng, IndependentProposal(model.q))
    state = @set sample.p = logdensity(model, sample)
    # TODO if (ever) using something else than MH & AnalyticalGibbs, each sampler might require it's own state.
    # sample, state
    state, state
end

"""
    step(sample, model, sampler, state)
Implementing the AbstractMCMC interface for steps given a state from the last step.
Cycles through the internal samplers using a random permutation
"""
function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, sampler::Gibbs, state::Sample)
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
    f::T
end

# AnalyticalGibbs is not able to propose a sample during the initial step
"""
    proposal(mh)
Get the proposal model of the Sampler.
"""
proposal(mh::AnalyticGibbs) = mh.f

"""
    proposal(mh)
Set the proposal model of the Sampler.
"""
set_proposal(mh::AnalyticGibbs, f::GibbsProposal{AnalyticProposal}) = @set mh.f = f


"""
    propose(m, s)
Propose a new sample for the AnalyticGibbs sampler.
"""
propose(rng::AbstractRNG, m::AnalyticGibbs, s) = propose(rng, m.f, s)

"""
    step(sample, model, sampler, state)
Implementing the AbstractMCMC interface for steps given a state from the last step.
AnalyticalGibbs always accepts the sample, since it is always the best possible sample given the prior sample
"""
function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, sampler::AnalyticGibbs, state::Sample)
    new_state = propose(rng, sampler, state)
    # sample, state
    new_state, new_state
end
