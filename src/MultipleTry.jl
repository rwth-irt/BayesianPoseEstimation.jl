# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    MultipleTry
Multiple Try Metropolis sampler (MTM).
Proposes multiple samples and selects one according to its importance weight → acceptance rate increases with the number of tries `n_tries`.
However, the standard implementation requires to propose `2*n_tries` to calculate auxiliary weights for the acceptance ratio.
"""
struct MultipleTry{Q,S} <: AbstractMCMC.AbstractSampler
    proposal::Q
    n_tries::Int64
    temp_schedule::S
end

"""
    IndependentMultipleTry
Proposes multiple samples and selects one according to its probability → acceptance rate increases with the number of tries `n_tries`.
If the proposal is independent from the previous sample, no auxiliary weights are requires and only `n_tries` are proposed.
"""
const IndependentMultipleTry = MultipleTry{<:Proposal{<:Any,typeof(propose_independent)}}

function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, sampler::MultipleTry)
    # TODO tempering
    # rand on PosteriorModel samples from prior in unconstrained domain
    sample = rand(model)
    # initial evaluation of the posterior logdensity
    sample = tempered_logdensity_sample(model, sample, 0.0)
    # sample, state are the same for MTM
    sample, MCMCState(sample, 0.0)
end

"""
    step(rng, model, sampler, state)
General MTM case without simplifications.
"""
function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, sampler::MultipleTry, old_state::MCMCState)
    # Schedule the likelihood tempering
    new_temp = increment_temperature(sampler.temp_schedule, old_state.temperature)
    # Propose N samples and calculate their importance weights
    pro_sample = propose(sampler.proposal, old_state.sample, sampler.n_tries)
    pro_sample = tempered_logdensity_sample(model, pro_sample, new_temp)
    pro_transition = transition_probability(sampler.proposal, pro_sample, old_state.sample)
    # TODO correct?
    pro_weights = logprobability(pro_sample) .- pro_transition

    # Select one sample proportional to its importance weight
    selected_index = gumbel_index(rng, pro_weights)
    selected_variables = select_variables_dim(variables(pro_sample), sampler.proposal, selected_index)
    selected = Sample(selected_variables, logprobability(pro_sample)[selected_index], loglikelihood(pro_sample)[selected_index])

    # Propose N-1 auxiliary variables samples
    aux_sample = propose(sampler.proposal, selected, sampler.n_tries - 1)
    aux_sample = tempered_logdensity_sample(model, aux_sample, new_temp)
    aux_transition = transition_probability(sampler.proposal, aux_sample, selected)
    aux_weights = logprobability(aux_sample) .- aux_transition
    # Sample from previous step is the N th auxiliary variables
    state_weight = logprobability(old_state.sample) - transition_probability(sampler.proposal, old_state.sample, selected)
    append!(aux_weights, state_weight)

    # acceptance ratio - sum in nominator and denominator
    α = logsumexp(pro_weights) - logsumexp(aux_weights)
    # MetropolisHastings acceptance
    if log(rand(rng)) > α
        # reject
        return old_state.sample, MCMCState(old_state.sample, new_temp)
    else
        # accept (always the case if difference positive since log([0,1])->[-inf,0])
        return selected, MCMCState(selected, new_temp)
    end
end

"""
    step(rng, model, sampler, state)
Simplification for independent proposals: I-MTM
"""
function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, sampler::IndependentMultipleTry, old_state::Sample)
    # Schedule the likelihood tempering
    new_temp = increment_temperature(sampler.temp_schedule, old_state.temperature)
    # Propose one sample via a kind of importance sampling
    pro_sample = propose(sampler.proposal, state, sampler.n_tries)
    pro_sample = tempered_logdensity_sample(model, pro_sample, new_temp)
    pro_transition = transition_probability(sampler.proposal, pro, state)
    proposed_weights = logprobability(pro_sample) .- pro_transition
    # First part of acceptance ratio
    α_nominator = logsumexp(proposed_weights)

    # Replace a sample according to its weight with the previous sample
    selected_index = gumbel_index(rng, proposed_weights)
    selected_vars = select_variables_dim(variables(pro), sampler.proposal, selected_index)
    selected = Sample(selected_vars, logprobability(pro_sample)[selected_index], loglikelihood(pro_sample)[selected_index])

    # From previous step, IndependentProposal so prev_sample can be anything
    state_weight = logprobability(old_state.sample) - transition_probability(sampler.proposal, old_state.sample, selected)
    proposed_weights[selected_index] = state_weight
    α_denominator = logsumexp(proposed_weights)

    # Acceptance ratio: Mind the log domain
    α = α_nominator - α_denominator
    # MetropolisHastings acceptance
    if log(rand(rng)) > α
        # reject
        return old_state.sample, MCMCState(old_state.sample, new_temp)
    else
        # accept (always the case if difference positive since log([0,1])->[-inf,0])
        return selected, MCMCState(selected, new_temp)
    end
end

"""
    gumbel_index(rng, log_weights)
Select an index from a categorical distribution ~ unnormalized `log_weights`.
The Gumbel-max trick is used to stay in the log domain and avoid numerical unstable exp or the more expensive log-sum-exp trick.
"""
gumbel_index(rng, log_weights) = argmax(log_weights .+ rand(rng, Gumbel(), size(log_weights)))


select_variables_dim(variables::NamedTuple, proposal::Proposal, index) = select_variables_dim(variables, updated_variables(proposal), index)

"""
    select_variables_dim(variables, graph_nodes, index)
For `variables` which contain multiple vectorized proposals.
Selects the `index` of the last dim only for `variables` in the graph.
"""
function select_variables_dim(variables::NamedTuple, ::Val{names}, index) where {names}
    # `..` is from EllipsisNotation.jl
    # WARN: Do not use view(x, .., index), since the GC can't collect the discarded dims which results in high GPU memory pressure → even more garbage collections or even out of memory exceptions.
    selected_vars = map(x -> x[.., index], variables[names])
    # return sample where the proposed variables are replaced
    merge(variables, selected_vars)
end
