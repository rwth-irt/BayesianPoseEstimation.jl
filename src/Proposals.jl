# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    Proposals.jl
Implement common proposal models with the convention of always proposing in the unconstrained domain ℝⁿ.
"""

"""
    evaluation_nodes(proposal, posterior)
Extract a SequentializedGraph of the nodes that need to be re-evaluated after proposing the new sample.
It contains all parents of the `proposal` nodes.
If the posterior is a PosteriorModel only the nodes of the prior are considered.
"""
evaluation_nodes(proposal::SequentializedGraph, posterior) = parents(posterior, values(proposal)...)
evaluation_nodes(proposal::SequentializedGraph, posterior::PosteriorModel) = parents(posterior.prior, proposal...)
evaluation_nodes(node::AbstractNode, posterior) = evaluation_nodes(sequentialize(node), posterior)

struct Proposal{names,F,G,M<:SequentializedGraph{names},E<:SequentializedGraph,B<:NamedTuple{names},C<:NamedTuple}
    propose_fn::F
    transition_probability_fn::G
    model::M
    evaluation::E
    proposal_bijectors::B
    posterior_bijectors::C
end

Base.show(io::IO, q::Proposal{names}) where {names} = print(io, "Proposal(names: $(names), $(q.propose_fn) & $(q.transition_probability_fn))")

Proposal(proposal_model, posterior_model, propose_fn, transition_probability_fn) = Proposal(propose_fn, transition_probability_fn, sequentialize(proposal_model), evaluation_nodes(proposal_model, posterior_model), map_materialize(bijector(proposal_model)), map_materialize(bijector(posterior_model)))

"""
    updated_variables(proposal)
Returns a Val{(names...,)} with all updated variables including the proposed and evaluated ones.
"""
updated_variables(::Proposal{names,<:Any,<:Any,<:Any,<:SequentializedGraph{evaluated}}) where {names,evaluated} = Val((names..., evaluated...))

# Convenience constructors

"""
    additive_proposal(proposal_model, posterior_model)
Propose a new sample from the previous by adding random values from the proposal distribution.
Does not use any simplifications so the support of the proposal distribution must be (-∞,∞). 
"""
additive_proposal(proposal_model, posterior_model) = Proposal(proposal_model, posterior_model, propose_additive, transition_probability_additive)

"""
    independent_proposal(proposal_model, posterior_model)
Propose samples independent from the previous one.
"""
independent_proposal(proposal_model, posterior_model) = Proposal(proposal_model, posterior_model, propose_independent, transition_probability_independent)

"""
    symmetric_proposal(proposal_model, posterior_model)
Propose a new sample from the previous one by using a symmetric proposal distribution.
Always returns 0 for the transition probability, as it cancels out in MH.
"""
symmetric_proposal(proposal_model, posterior_model) = Proposal(proposal_model, posterior_model, propose_additive, transition_probability_symmetric)

# Interface methods

"""
    propose(proposal, previous_sample, [dims...])
Generate a new sample using the `proposal` and maybe conditioning on the old `sample`.
Use dims to sample propose the variables multiple times (vectorization support).
"""
propose(proposal::Proposal, previous_sample, dims...) = proposal.propose_fn(proposal, previous_sample, dims...)

"""
    transition_probability(proposal, new_sample, prev_sample)
The probability of transitioning from the `prev_sample` to the `new_sample` given the `proposal` model.
"""
transition_probability(proposal::Proposal, new_sample, previous_sample) = proposal.transition_probability_fn(proposal, new_sample, previous_sample)

# propose_fn implementations

"""
    propose(proposal, previous_sample, [dims...])
Propose a new sample by adding a random perturbation from the proposal model.
"""
function propose_additive(proposal::Proposal, previous_sample, dims...)
    # Propose in unconstrained domain
    proposed = previous_sample ⊕ rand(proposal.model, dims...)
    # Evaluate in model domain
    model_sample, _ = to_model_domain(proposed, proposal.posterior_bijectors)
    evaluated = evaluate(proposal.evaluation, variables(model_sample))
    # Sampling in unconstrained domain
    to_unconstrained_domain(Sample(evaluated), proposal.posterior_bijectors)
end

"""
    propose(proposal, previous_sample, [dims...])
Independent samples are simply random values from the model.
"""
function propose_independent(proposal::Proposal, previous_sample, dims...)
    # Propose in the model domain
    model_sample, _ = to_model_domain(previous_sample, proposal.posterior_bijectors)
    proposed = merge(model_sample, rand(proposal.model, dims...))
    evaluated = evaluate(proposal.evaluation, variables(proposed))
    # Sampling in unconstrained domain, proposals might have other bijectors than the posterior (prior) model
    merged_bijectors = merge(proposal.posterior_bijectors, proposal.proposal_bijectors)
    to_unconstrained_domain(Sample(evaluated), merged_bijectors)
end

# transition_probability_fn implementations

"""
    transition_probability_additive(proposal, new_sample, prev_sample)
For the general case of additive proposals, where the forward and backward transition probabilities do not cancel out.
"""
transition_probability_additive(proposal::Proposal{names}, new_sample, previous_sample) where {names} = logdensityof(proposal.model, variables(new_sample[Val(names)] ⊖ previous_sample[Val(names)]))

"""
    transition_probability_independent(proposal, new_sample, prev_sample)
For independent proposals, the transition probability does not depend on the previous sample.
Since the proposal model might be defined in a constrained domain, the sample is transformed and the logjac adjustment added to the logdensity.
"""
function transition_probability_independent(proposal::Proposal{names}, new_sample, previous_sample) where {names}
    # Evaluate only the bijector of the proposed variables to avoid adding the logjac correction for variables which are not evaluated in logdensityof
    model_sample, logjac = to_model_domain(new_sample[Val(names)], proposal.proposal_bijectors)
    logdensityof(proposal.model, variables(model_sample)) + logjac
end

"""
    transition_probability_symmetric(proposal, new_sample, prev_sample)
For symmetric proposals, the forward and backward transition probability cancels out in MH - return 0.
"""
transition_probability_symmetric(proposal::Proposal, new_sample, previous_sample) = 0

