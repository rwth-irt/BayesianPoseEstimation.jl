# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    AbstractProposal
Has a `model` which supports `rand(rng, model, dims...)` and `logdensityof(model, x)`.
"""
abstract type AbstractProposal end

"""
    model(proposal)
Get the internal probabilistic model from which the proposals are generated.
"""
model(proposal::AbstractProposal) = proposal.model

"""
    transition_probability(proposal, new_sample, prev_sample)
For the general case of dependent samples for a previous and the new `Sample`.
"""
transition_probability(proposal::AbstractProposal, new_sample, prev_sample) = logdensityof(model(proposal), new_sample - prev_sample)

# SymmetricProposal

"""
    SymmetricProposal
Propose samples from the previous one by using a symmetric proposal distribution.
"""
struct SymmetricProposal{T} <: AbstractProposal
    model::T

    function SymmetricProposal(model::T) where {T}
        if !is_identity(model)
            throw(DomainError(model, "Model domain is not is not defined on ℝᴺ"))
        end
        new{T}(model)
    end
end

"""
    propose(rng, proposal, [sample], [dims...])
Generate a new sample using the `proposal` and maybe conditioning on the old `sample`.
Use dims to sample propose the variables multiple times (vectorization support).
"""
propose(rng::AbstractRNG, proposal::SymmetricProposal, sample::Sample, dims...) = sample + rand(rng, model(proposal), dims...)

"""
    transition_probability(proposal, new_sample, prev_sample)
For symmetric proposals, the forward and backward transition probability cancels out
"""
transition_probability(proposal::SymmetricProposal, new_sample::Sample, ::Sample) = 0.0

# IndependentProposal

"""
    IndependentProposal
Propose samples independent from the previous one.
"""
struct IndependentProposal{T} <: AbstractProposal
    model::T
end

# Only makes sense for IndependentProposal
Base.rand(rng::AbstractRNG, proposal::IndependentProposal, dims::Integer...) = rand(rng, model(proposal), dims...)

"""
    propose(rng, proposal, sample, [dims...])
Independent samples are just random values from the model.
"""
propose(rng::AbstractRNG, proposal::IndependentProposal, sample::Sample, dims...) = merge(sample, rand(rng, proposal, dims...))

"""
    transition_probability(proposal, new_sample, prev_sample)
For independent proposals, the transition probability does not depend on the previous sample.
"""
transition_probability(proposal::IndependentProposal, new_sample::Sample, ::Sample) = logdensityof(model(proposal), new_sample)

# GibbsProposal

"""
    GibbsProposal
Has a function `fn(sample::Sample)` which analytically samples from a distribution conditioned on `sample`.
"""
struct GibbsProposal{T} <: AbstractProposal
    fn::T
end

"""
    propose([rng], proposal, sample, [dims...])
Analytic proposals are conditioned on the other variables of the previous sample.
A subset of the variables is proposed by the internal proposal model and merged with the previous sample.
Since sampling is analytic, `dims` solely depends on the previous sample and the `rng` is not used.
"""
function propose(::AbstractRNG, proposal::GibbsProposal, sample::Sample, dims...)
    gibbs_sample = proposal.fn(sample)
    merge(sample, gibbs_sample)
end

"""
    transition_probability(proposal, new_sample, prev_sample)
Gibbs proposals are always accepted so `Inf` is returned.
"""
transition_probability(::GibbsProposal, ::Sample, ::Sample) = Inf

# TODO Custom quaternion proposal: composition is the quaternion product and normalize