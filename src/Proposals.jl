# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    Proposals.jl
Implement common proposal models with the convention of always proposing in the unconstrained domain ℝⁿ.
"""

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
    bijector(proposal)
Get a named tuple of bijectors for the proposal. Assuming that a proposals live in ℝⁿ, the default bijector is empty.
"""
Bijectors.bijector(::AbstractProposal) = (;)

"""
    transition_probability(proposal, new_sample, prev_sample)
For the general case of dependent samples for a previous and the new `Sample`.
Since the proposal model might be defined in a constrained domain, the sample is transformed and the logjac adjustment added to the logdensity.
"""
function transition_probability(proposal::AbstractProposal, new_sample, prev_sample)
    diff_sample = new_sample - prev_sample
    model_sample, logjac = to_model_domain(diff_sample, proposal.bijectors)
    logdensityof(model(proposal), model_sample) + logjac
end

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
struct IndependentProposal{T,B} <: AbstractProposal
    model::T
    bijectors::B
end

IndependentProposal(model) = IndependentProposal(model, map_materialize(bijector(model)))

"""
    bijector(proposal)
Get a named tuple of bijectors for the proposal.
Independent proposals might be constrained.
"""
Bijectors.bijector(proposal::IndependentProposal) = proposal.bijectors

"""
    rand(rng, proposal, dims...)
Generate a random sample from the proposal.
Only makes sense for independent proposals, since they do not require any prior sample.
Per convention, the generated sample is transformed to ℝⁿ.
"""
Base.rand(rng::AbstractRNG, proposal::IndependentProposal, dims::Integer...) = to_unconstrained_domain(rand(rng, model(proposal), dims...), proposal.bijectors)

"""
    propose(rng, proposal, sample, [dims...])
Independent samples are just random values from the model.
"""
propose(rng::AbstractRNG, proposal::IndependentProposal, sample::Sample, dims...) = merge(sample, rand(rng, proposal, dims...))

"""
    transition_probability(proposal, new_sample, prev_sample)
For independent proposals, the transition probability does not depend on the previous sample.
Since the proposal model might be defined in a constrained domain, the sample is transformed and the logjac adjustment added to the logdensity.
"""
function transition_probability(proposal::IndependentProposal, new_sample::Sample, ::Sample)
    model_sample, logjac = to_model_domain(new_sample, proposal.bijectors)
    logdensityof(model(proposal), model_sample) + logjac
end

# TODO Custom quaternion proposal: composition is the quaternion product and normalize
