# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using AbstractMCMC
using Accessors
using LogExpFunctions
using Random

"""
    SequentialMonteCarlo
Sequential Monte Carlo with systematic resampling and 
"""
struct SequentialMonteCarlo{Q<:Proposal,K<:Function,F<:Function,S} <: AbstractMCMC.AbstractSampler
    # TODO In this case the symmetric simplification does not hold anymore, use AdditiveProposal
    forward_kernel::K
    weight_increment::F
    temp_scheduler::S
    n_particles::Int64
    log_resampling_threshold::Float64
end

struct SmcState{S<:Sample}
    # Contains multiple values for the variables
    sample::S
    log_weights::Vector{Float64}
    log_evidence::Float64
    temperature::Float64
    # TODO likelihood tempering
end


function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, sampler::SequentialMonteCarlo)
    # NOTE This is an IS step
    # rand on PosteriorModel samples from prior in unconstrained domain
    s = rand(model, sampler.n_particles)
    # tempering starts with ϕ₀=0 
    s = set_logp(s, logdensityof(model, s, 0))
    # ϕ₀=0 → importance distribution = target density → wᵢ=1, normalized:
    log_weights = fill(-log(sampler.n_particles), sampler.n_particles)
    # IS normalizing constant: 1/n * ∑ₙ wᵢ = 1/n_particles * n_particles = 1 → log(1) = 0
    state = SmcState(s, log_weights, 0.0, 0.0)
    # sample, state are the same for MH
    state, state
end

"""
    step(rng, model, sampler, state)
Generic SMC sampler according to 3.1.1. (Sequential Monte Carol Samplers, Del Moral 2006)
"""
function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, sampler::SequentialMonteCarlo, old_state::SmcState)
    # TODO Does a mutable SmcState make sense?
    # Schedule the likelihood tempering
    new_temp = old_state.temperature + increment_temperature(sampler.temp_scheduler, old_state.temperature)
    new_state = @set old_state.temperature = new_temp
    # Draw new particles using the forward kernel
    new_sample = propose(sampler.forward_kernel, old_state.sample, sampler.n_particles)
    new_sample = set_logp(new_sample.sample, logdensityof(model, new_sample, new_temp))
    new_sample = forward(sampler.kernel, new_sample, old_state.sample)
    # Update weights using backward kernel
    new_state = @set new_state.sample = new_sample
    unnormalized = old_state.log_weights .+ incremental_weights(sampler.kernel, new_state, old_state)
    new_state = @set new_state.log_weights = normalize_log_weights!(unnormalized)
    # Increment the log-evidence using the unnormalized weights
    log_evidence = old_state.log_evidence + logsumexp(incr_weights)
    new_state = @set new_state.log_evidence = log_evidence
    # TODO return state twice or take the overhead of selecting a single value? Which one? Max. weight or randomly according weights?
    resampled = maybe_resample(new_state)
    resampled, resampled
end

# SmcKernels, implement:
# proposal(kernel): Getter for the proposal
# forward(kernel, new_sample, old_sample): forward kernel for the proposed sample
# incremental_weights(kernel, new_state, old_state): calculate the unnormalized incremental weights

propose(kernel, previous_sample, n_particles) = @set old_state.sample = propose(proposal(kernel), previous_sample, n_particles)

"""
    ForwardProposalKernel(proposal)
Use the transition probability of the forward proposal for the backward L-kernel.
This results in a weight increment similar to a Metropolis-Hastings acceptance ratio.
(Increasing the efficiency of Sequential Monte Carlo samplers through the use of approximately optimal L-kernels, Green 2022)
"""
struct ForwardProposalKernel{Q,R}
    proposal::Q
end

proposal(kernel::ForwardProposalKernel) = kernel.proposal
forward(kernel::ForwardProposalKernel, new_sample, old_sample) = new_sample

"""
    increment_weights(kernel, new_state, old_state)
Calculate the unnormalized incremental log using a "forward proposal L-kernel" (Increasing the Efficiency of Sequential Monte Carlo Samplers..., Green 2022).
The weights are updated similarly to a Metropolis-Hastings acceptance ratio.
"""
function incremental_weights(kernel::ForwardProposalKernel, new_state::SmcState, old_state::SmcState)
    forward = transition_probability(kernel.proposal, new_state.sample, old_state.sample)
    backward = transition_probability(kernel.proposal, old_state.sample, new_state.sample)
    logprob(new_state.sample) + backward - logprob(old_state.sample) - forward + state.log_weights
end


struct MhKernel{Q,R}
    proposal::Q
    rng::R
end

proposal(kernel::MhKernel) = kernel.proposal
forward(kernel::MhKernel, new_sample, old_sample) = mh_kernel!(kernel.rng, proposal(kernel), new_sample, old_sample)

"""
    increment_weights(kernel, new_state, old_state)
Calculate the unnormalized incremental log using an MCMC Kernel (Sequential Monte Carlo Samplers, Del Moral 2006).
For a likelihood tempered target γ = p(z|θ)ᵠp(θ) the formula simplifies to γ₂/γ₁ = p(z/θ₁)^(ϕ₂ - ϕ₁) (Efficient Sequential Monte-Carlo Samplers for Bayesian Inference, Nguyen 2016)
"""
incremental_weights(::MhKernel, new_state::SmcState, old_state::SmcState) = (new_state.temperature - old_state.temperature) .* logprob(old_state.sample)

"""
normalize_log_weights(log_weights)
    Normalization of the weights in the log domain using the log-sum-exp trick.
"""
normalize_log_weights(log_weights) = log_weights .- logsumexp(log_weights)

"""
    maybe_resample(state, log_threshold)
Resample the variables of the `state` with their respective log-weights & -probabilities if the log effective sample size is smaller than the `log_threshold`
"""
function maybe_resample(state::SmcState, log_threshold)
    if effective_sample_size(state.log_weights) < log_threshold
        # Resample variables
        indices = systematic_resampling_indices(state.log_weights)
        vars = map(x -> select_var_dim(x, indices), variables(state.sample))
        log_probs = logprob(state.sample)[indices]
        re_sample = Sample(vars, log_probs)
        # Reset weights
        log_weights = fill(-log(length(log_weights)), length(state.log_weights))
        SmcState(re_sample, log_weights, log_probs)
    else
        state
    end
end

"""
    effective_sample_size(log_weights)
Expects normalized log_weights and calculates log(ESS)=log(1/∑(wᵢ²))=-log(∑exp(2*log(wᵢ)))
"""
effective_sample_size(log_weights) = -logsumexp(2 .* log_weights)

"""
    systematic_resampling_indices(rng, log_weights)
Expects normalized `log_weights` and returns indices distributed according to the probability mass function of the weights. 
"""
function systematic_resampling_indices(rng::AbstractRNG, log_weights::AbstractArray)
    Nₚ = length(log_weights)
    # cumulative
    log_c = first(log_weights)
    # starting point
    r = rand(rng, Uniform(0.0, 1.0 / Nₚ))
    # current and resampled indices
    i = 1
    res = similar(log_weights, Int64)
    for n in 1:Nₚ
        # Julia: n-1 so we only add (Nₚ-1) steps
        U = log(r + (n - 1) / Nₚ)
        while U > log_c
            i += 1
            log_c = logaddexp(log_c, log_weights[i])
        end
        res[n] = i
    end
    res
end
