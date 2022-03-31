# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using MeasureTheory, Soss
using Random

"""
    AbstractProposal
Has a `model` which supports rand() and logdensity
"""
abstract type AbstractProposal end

# TODO probably no use? Might even cause undefined behaviour
# """
#     Proposal
# Propose samples from the previous one by using a general proposal distribution.
# """
# struct Proposal{T<:AbstractMeasure} <: AbstractProposal
#     model::T

#     function Proposal(m::T) where {T<:Soss.AbstractModel}
#         # Generally only unconstrained proposal models are supported
#         # E.g. an exponential proposal would result in s - c_cond < 0 and thus an invalid logdensity
#         if !is_identity(xform(m | (;)))
#             throw(DomainError(m, "Model is not unconstrained, i.e. requires maps to a domain which is not ℝ"))
#         end
#         new{T}(m)
#     end
# end

"""
    model(q)
Get the internal probabilistic model from which the proposals are generated.
"""
model(q::AbstractProposal) = q.model

"""
    rand(rng, q)
Generate a new raw state `θ` using the `model` of the proposal `q`.
Whether the sample is Constrained or not is determined 
"""
Base.rand(rng::AbstractRNG, q::AbstractProposal) = rand(rng, model(q))

"""
    rand(q)
Generate a new raw state `θ` using the `model` of the proposal `q`.
"""
Base.rand(q::AbstractProposal) = rand(Random.GLOBAL_RNG, q)

"""
    propose(rng, q, s)
For the general case of dependent samples.
Support for Gibbs: applies the current state to the args of the model.
"""
propose(rng::AbstractRNG, q::AbstractProposal, s::Sample) = s + rand(rng, model(q)(state(s)))

"""
    propose(q, s)
For the general case of dependent samples.
"""
propose(q::AbstractProposal, s::Sample) = propose(Random.GLOBAL_RNG, q, s)

"""
    propose(q, s)
For an initial sample, not supported by every Proposal type.
Mainly for dispatch of a missing random number generator.
"""
propose(q::AbstractProposal) = propose(Random.GLOBAL_RNG, q)

"""
    transition_probability(q, s, s_cond)
For the general case of dependent samples.
Support for Gibbs: applies the previous state to the args of the model.
"""
# The proposal model had s_cond applied to args
# Dependent samples are sampled in the unconstrained domain -> raw state for conditional
transition_probability(q::AbstractProposal, s::Sample, s_cond::Sample) = logdensity(model(q)(state(s_cond)), raw_state(s - s_cond))

# SymmetricProposal


"""
    SymmetricProposal
Propose samples from the previous one by using a symmetric proposal distribution.
"""
struct SymmetricProposal{T<:AbstractMeasure} <: AbstractProposal
    model::T

    function SymmetricProposal(m::T) where {T<:AbstractMeasure}
        if !is_identity(as(m))
            throw(DomainError(m, "Model is not unconstrained, i.e. requires maps to a domain which is not ℝ"))
        end
        new{T}(m)
    end

    function SymmetricProposal(m::T) where {T<:Soss.AbstractModel}
        # TODO workaround for Gibbs which moves variables to the arguments
        if isempty(arguments(m)) && !is_identity(xform(m | (;)))
            throw(DomainError(m, "Model is not unconstrained, i.e. requires maps to a domain which is not ℝ"))
        end
        new{T}(m)
    end
end

"""
    transition_probability(q, s, s_cond)
For symmetric proposals, the forward and backward transition probability cancels out
"""
transition_probability(q::SymmetricProposal, ::Sample, ::Sample) = 0.0

# IndependentProposal

"""
    IndependentProposal
Propose samples independent from the previous one.
"""
struct IndependentProposal{T<:AbstractMeasure} <: AbstractProposal
    model::T
end

"""
    propose(rng, q)
Independent samples are just random values from the model.
"""
propose(rng::AbstractRNG, q::IndependentProposal) = Sample(rng, model(q))

"""
    propose(rng, q, s)
Independent samples are just random values from the model.
Support for Gibbs: applies the current state to the args of the model.
"""
propose(rng::AbstractRNG, q::IndependentProposal, s::Sample) = Sample(rng, model(q)(state(s)))

"""
    transition_probability(q, s, s_cond)
For independent proposals, the transition probability does not depend on the previous sample.
Support for Gibbs: applies the current state to the args of the model.
"""
# IndependentProposal: logdensity in constrained domain -> transformation via state
transition_probability(q::IndependentProposal, s::Sample, s_cond::Sample) = logdensity(model(q)(state(s_cond)), state(s))

# GibbsProposal


# TODO  Do not depend on Soss anymore. Either all Models must support conditioning like applying args to a Soss model or we need a random(rng, model, sample) function. logdensity(model, sample) should work out of the box.
"""
    GibbsProposal
Propose only a subset of the variables.
Thus, GibbsProposal is only an overlay over another AbstractProposal type `Q` which uses a conditional `model` for its proposal.
"""
struct GibbsProposal{Q<:AbstractProposal} <: AbstractProposal
    internal_proposal::Q
end

"""
    GibbsProposal(model, var)
Convenience constructor which automatically transforms the `model`` so only the variables `var` will be proposed.
Since Gibbs assumes all other variables to be known, the required dependencies are moved to the `args` of the model.
"""
function GibbsProposal{Q}(model::AbstractMeasure, var::Symbol...) where {Q<:AbstractProposal}
    # Sanity check is only executed without arguments
    SymmetricProposal(model)
    # Move var to arguments & create GibbsProposal
    Soss.likelihood(Model(model), var...)(argvals(model)) |> Q |> GibbsProposal
end

"""
    model(q)
Get the probabilistic model of the internal proposal model from which the proposals are generated.
"""
model(q::GibbsProposal) = model(q.internal_proposal)

"""
    propose(rng, q, s)
Gibbs proposals are conditioned on the other variables of the previous sample.
A subset of the variables is proposed by the internal proposal model and merged with the previous sample.
"""
function propose(rng::AbstractRNG, q::GibbsProposal, s::Sample)
    s2 = propose(rng, q.internal_proposal, s)
    merge(s, s2)
end

"""
    transition_probability(q, s, s_cond)
Forwards it to the transition probability model of the intern proposal distribution.
"""
transition_probability(q::GibbsProposal, s::Sample, s_cond::Sample) = transition_probability(q.internal_proposal, s, s_cond)

# AnalyticProposal

"""
    AnalyticProposal
Has a callable `f(s)` which returns a new Sample for a subset of the model variables given a previous sample `s`.
"""
struct AnalyticProposal{T} <: AbstractProposal
    f::T
end

"""
    propose(rng, q, s)
Analytic proposals are conditioned on the other variables of the previous sample.
A subset of the variables is proposed by the internal proposal model and merged with the previous sample.
"""
function propose(::AbstractRNG, q::AnalyticProposal{Q}, s::Sample) where {Q}
    s2 = q.f(s)
    merge(s, s2)
end

"""
    transition_probability(q, s, s_cond)
Analytic proposals are always accepted
"""
transition_probability(::AnalyticProposal, s::Sample, s_cond::Sample) = Inf
