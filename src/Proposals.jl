# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    Proposals.jl
Implement common proposal models with the convention of always proposing in the unconstrained domain ℝⁿ.
"""

"""
    proposal(T, proposal_model, posterior_model)
Helps with the correct construction of some proposal type `T(proposal_model, evaluation, bijectors)`.
"""
proposal(::Type{T}, proposal_model, posterior_model) where {T} = T(proposal_model, evaluation_nodes(proposal_model, posterior_model), map_materialize(bijector(proposal_model)))

"""
    evaluation_nodes(proposal, posterior)
Extract a SequentializedGraph of the nodes that need to be re-evaluated after proposing the new sample.
It contains all parents of the `proposal` nodes in the prior of the `posterior` node.
"""
function evaluation_nodes(proposal::SequentializedGraph, posterior::AbstractNode{name}) where {name}
    p = parents(posterior, values(proposal)...)
    Base.structdiff(p, (; name => ()))
end
evaluation_nodes(proposal_model::AbstractNode, posterior_model::AbstractNode) = evaluation_nodes(sequentialize(proposal_model), posterior_model)

"""
    SymmetricProposal
Propose a new sample from the previous one by using a symmetric proposal distribution.
"""
struct SymmetricProposal{M,E,B}
    model::M
    evaluation::E
    bijectors::B
end
SymmetricProposal(proposal_model, posterior_model) = proposal(SymmetricProposal, proposal_model, posterior_model)

"""
    propose(proposal, [sample, dims...])
Generate a new sample using the `proposal` and maybe conditioning on the old `sample`.
Use dims to sample propose the variables multiple times (vectorization support).
"""
function propose(proposal::SymmetricProposal, sample::Sample, dims...)
    # Propose in unconstrained domain
    proposed = sample + rand(proposal.model, dims...)
    # Evaluate in model domain
    model_sample, _ = to_model_domain(proposed, proposal.bijectors)
    evaluated = evaluate(proposal.evaluation, variables(model_sample))
    # Sampling in unconstrained domain
    to_unconstrained_domain(Sample(evaluated), proposal.bijectors)
end

"""
    transition_probability(proposal, new_sample, prev_sample)
For symmetric proposals, the forward and backward transition probability cancels out.
Will be combined with prior and likelihood by broadcasting, so returning a scalar is enough.
"""
transition_probability(proposal::SymmetricProposal, new_sample::Sample, ::Sample) = 0

# IndependentProposal

"""
    IndependentProposal
Propose samples independent from the previous one.
"""
struct IndependentProposal{M,E,B}
    model::M
    evaluation::E
    bijectors::B
end
IndependentProposal(proposal_model, posterior_model) = proposal(IndependentProposal, proposal_model, posterior_model)


"""
    rand(proposal, dims...)
Generate a random sample from the proposal.
Only makes sense for independent proposals, since they do not require any prior sample.
Per convention, the generated sample is transformed to ℝⁿ.
"""
function Base.rand(proposal::IndependentProposal, dims::Integer...)
    proposed = rand(proposal.model, dims...) |> Sample
    to_unconstrained_domain(proposed, proposal.bijectors)
end

"""
    propose(proposal, sample, [dims...])
Independent samples are just random values from the model.
"""
function propose(proposal::IndependentProposal, sample::Sample, dims...)
    # Propose in the model domain
    model_sample, _ = to_model_domain(sample, proposal.bijectors)
    proposed = merge(model_sample, rand(proposal.model, dims...))
    evaluated = evaluate(proposal.evaluation, variables(proposed))
    # Sampling in unconstrained domain
    to_unconstrained_domain(Sample(evaluated), proposal.bijectors)
end

"""
    transition_probability(proposal, new_sample, prev_sample)
For independent proposals, the transition probability does not depend on the previous sample.
Since the proposal model might be defined in a constrained domain, the sample is transformed and the logjac adjustment added to the logdensity.
"""
function transition_probability(proposal::IndependentProposal, new_sample::Sample, ::Sample)
    model_sample, logjac = to_model_domain(new_sample, model_bijectors(proposal))
    logdensityof(proposal.model, variables(model_sample)) + logjac
end

# Evaluate only the bijector of the proposals model variables to avoid adding the logjac correction for variables which are not evaluated in logdensityof
model_bijectors(proposal::IndependentProposal{<:AbstractNode{name}}) where {name} = proposal.bijectors[(name,)]
model_bijectors(proposal::IndependentProposal{<:SequentializedGraph{names}}) where {names} = proposal.bijectors[names]
