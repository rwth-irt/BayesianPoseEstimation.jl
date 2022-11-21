# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    Proposals.jl
Implement common proposal models with the convention of always proposing in the unconstrained domain ℝⁿ.
"""

# TODO Maybe implement specific behaviors via traits? E.g. all symmetric proposals should return 0
# SymmetricProposal
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
struct SymmetricProposal{T,U}
    model::T
    evaluation::U

    function SymmetricProposal(proposal_model::T, posterior_model) where {T}
        evaluation_model = evaluation_nodes(proposal_model, posterior_model)
        new{T,typeof(evaluation_model)}(proposal_model, evaluation_model)
    end
end

"""
    propose(proposal, [sample], [dims...])
Generate a new sample using the `proposal` and maybe conditioning on the old `sample`.
Use dims to sample propose the variables multiple times (vectorization support).
"""
function propose(proposal::SymmetricProposal, sample::Sample, dims...)
    # proposal step
    proposed = sample + rand(proposal.model, dims...)
    # determinstic evaluation step
    Sample(evaluate(proposal.evaluation, variables(proposed)))
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
struct IndependentProposal{T,U,B}
    model::T
    evaluation::U
    bijectors::B
end

IndependentProposal(proposal_model, posterior_model) = IndependentProposal(proposal_model, evaluation_nodes(proposal_model, posterior_model), map_materialize(bijector(proposal_model)))

"""
    bijector(proposal)
Get a named tuple of bijectors for the proposal.
Independent proposals might be constrained.
"""
Bijectors.bijector(proposal::IndependentProposal) = proposal.bijectors

"""
    rand(proposal, dims...)
Generate a random sample from the proposal.
Only makes sense for independent proposals, since they do not require any prior sample.
Per convention, the generated sample is transformed to ℝⁿ.
"""
function Base.rand(proposal::IndependentProposal, dims::Integer...)
    proposed = rand(proposal.model, dims...) |> Sample
    to_unconstrained_domain(proposed, bijector(proposal))
end

"""
    propose(proposal, sample, [dims...])
Independent samples are just random values from the model.
"""
function propose(proposal::IndependentProposal, sample::Sample, dims...)
    # proposal step
    proposed = merge(sample, rand(proposal, dims...))
    # determinstic evaluation step
    Sample(evaluate(proposal.evaluation, variables(proposed)))
end

"""
    transition_probability(proposal, new_sample, prev_sample)
For independent proposals, the transition probability does not depend on the previous sample.
Since the proposal model might be defined in a constrained domain, the sample is transformed and the logjac adjustment added to the logdensity.
"""
function transition_probability(proposal::IndependentProposal, new_sample::Sample, ::Sample)
    model_sample, logjac = to_model_domain(new_sample, bijector(proposal))
    logdensityof(proposal.model, variables(model_sample)) + logjac
end
