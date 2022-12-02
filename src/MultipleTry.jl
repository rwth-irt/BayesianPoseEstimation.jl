# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# TODO logdensity interface idea: Simply add tiles to parameters
# logdensity(model, samples, tiles)

using AbstractMCMC
using LogExpFunctions
using Random

"""
    MultipleTry
TODO dispatch based on proposal type
"""
struct MultipleTry{Q} <: AbstractMCMC.AbstractSampler
    # TODO In this case the symmetric simplification does not hold anymore, use AdditiveProposal
    proposal::Q
    n_tries::Int64
end


function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, sampler::MultipleTry)
    # rand on PosteriorModel samples from prior in unconstrained domain
    prior_sample = rand(model)
    # initial evaluation of the posterior logdensity
    ℓ_sample = Sample(variables(prior_sample), logdensityof(model, prior_sample))
    # sample, state are the same for MH
    ℓ_sample, ℓ_sample
end

"""
    step(rng, model, sampler, state)
General MTM case without independence simplifications.
"""
function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, sampler::MultipleTry, state::Sample)
    # Mind the log domain
    # Propose one sample via a kind of importance sampling
    proposed = propose(sampler.proposal, state, sampler.n_tries)
    ℓ_model = logdensityof(model, proposed)
    ℓ_transition = transition_probability(sampler.proposal, proposed, state)
    proposed_weights = ℓ_model .- ℓ_transition

    selected_index = gumbel_index(rng, proposed_weights)
    selected_vars = select_variables_dim(proposed, variables(n_proposal), selected_index)
    selected = Sample(selected_vars, ℓ_model[selected_index])

    # Propose the N-1 auxiliary variables samples
    auxiliary = propose(sampler.proposal, selected, sampler.n_tries - 1)
    auxiliary_weights = logdensityof(model, selected) .- transition_probability(sampler.proposal, auxiliary, selected)
    # Sample from previous step is the N th auxiliary variables
    state_weight = logprob(state) - transition_probability(sampler.proposal, state, selected)
    append!(auxiliary_weights, state_weight)

    # acceptance ratio - sum in nominator and denominator
    α = logsumexp(proposed_weights) - logsumexp(auxiliary_weights)
    # MetropolisHastings acceptance
    if log(rand(rng)) > α
        # reject
        return state, state
    else
        # accept (always the case if difference positive since log([0,1])->[-inf,0])
        return selected, selected
    end
end

"""
    step(rng, model, sampler, state)
Simplification for independent proposals: I-MTM
"""
function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, sampler::MultipleTry{<:IndependentProposal}, state::Sample)
    # Propose one sample via a kind of importance sampling
    proposed = propose(sampler.proposal, state, sampler.n_tries)
    ℓ_model = logdensityof(model, proposed)
    ℓ_transition = transition_probability(sampler.proposal, proposed, state)
    proposed_weights = ℓ_model .- ℓ_transition
    # First part of acceptance ratio
    α_nominator = logsumexp(proposed_weights)

    # Replace a sample according to its weight with the previous sample
    selected_index = gumbel_index(rng, proposed_weights)
    selected_vars = select_variables_dim(proposed, variables(n_proposal), selected_index)
    selected = Sample(selected_vars, ℓ_model[selected_index])
    # From previous step, IndependentProposal so prev_sample can be anything
    state_weight = logprob(state) - transition_probability(sampler.proposal, state, selected)
    proposed_weights[selected_index] = state_weight
    α_denominator = logsumexp(proposed_weights)

    # Acceptance ratio: Mind the log domain
    α = α_nominator - α_denominator
    # MetropolisHastings acceptance
    if log(rand(rng)) > α
        # reject
        return state, state
    else
        # accept (always the case if difference positive since log([0,1])->[-inf,0])
        return selected, selected
    end
end

"""
    gumbel_index(rng, log_weights)
Select an index from a categorical distribution ~ unnormalized `log_weights`.
The Gumbel-max trick is used to stay in the log domain and avoid numerical unstable exp or the more expensive log-sum-exp trick.
"""
gumbel_index(rng, log_weights) = argmax(log_weights .+ rand(rng, Gumbel(), size(log_weights)))

"""
    select_variables_dim(variables, proposal, index)
For `variables` which contain multiple vectorized proposals.
Selects the `index` of the last dim only for `variables` from the `proposal`.
"""
function select_variables_dim(variables, proposal, index)
    # TODO interface functions for proposal model & evaluation
    var_names = proposal_varnames(proposal.model, proposal.evaluation)
    selected_vars = map(x -> selectdim(x, ndims(x), index), variables[var_names])
    # return sample where the proposed variables are replaced
    merge(variables, selected_vars)
end
