# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using AbstractMCMC
using LogExpFunctions
using Random

"""
    SequentialMonteCarlo
Sequential Monte Carlo with systematic resampling and likelihood tempering via p(θ|z) ∝ p(z|θ)ᶲ p(θ).
"""
struct SequentialMonteCarlo{K,S} <: AbstractMCMC.AbstractSampler
    kernel::K
    temp_scheduler::S
    n_particles::Int64
    log_resample_threshold::Float64
end

struct SmcState{S<:Sample,W<:AbstractVector,L<:AbstractVector}
    # Contains multiple values for the variables
    sample::S
    log_weights::W
    log_likelihood::L
    log_evidence::Float64
    temperature::Float64
end

function tempered_logdensity(log_prior, log_likelihood, temp=1)
    if temp == 0
        return log_prior
    end
    if temp == 1
        return add_logdensity(log_prior, log_likelihood)
    end
    add_logdensity(log_prior, temp .* log_likelihood)
end

function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, sampler::SequentialMonteCarlo)
    # NOTE This is an IS step
    # rand on PosteriorModel samples from prior in unconstrained domain
    s = rand(model, sampler.n_particles)
    # tempering starts with ϕ₀=0
    log_prior, log_likelihood = prior_and_likelihood(model, s)
    log_full = tempered_logdensity(log_prior, log_likelihood, 0)
    s = set_logp(s, log_full)

    # ϕ₀=0 → importance distribution = target density → wᵢ=1, normalized:
    normalized_log_weights = fill(-log(sampler.n_particles), sampler.n_particles)
    # IS normalizing constant: 1/n * ∑ₙ wᵢ = n_particles / n_particles = 1 → log(1) = 0
    state = SmcState(s, normalized_log_weights, log_likelihood, 0.0, 0.0)

    state.sample, state
end

"""
    step(rng, model, sampler, state)
Generic SMC sampler according to 3.1.1. (Sequential Monte Carol Samplers, Del Moral 2006)
"""
function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, sampler::SequentialMonteCarlo, old_state::SmcState)
    # Schedule the likelihood tempering
    new_temp = increment_temperature(sampler.temp_scheduler, old_state.temperature)

    # Draw new particles using the forward kernel
    proposed_sample = propose(sampler.kernel, old_state.sample, sampler.n_particles)
    log_prior, log_likelihood = prior_and_likelihood(model, proposed_sample)
    log_full = tempered_logdensity(log_prior, log_likelihood, new_temp)
    proposed_sample = set_logp(proposed_sample, log_full)
    new_sample = forward(sampler.kernel, proposed_sample, old_state.sample)

    # Update weights using backward kernel
    incr_weights = incremental_weights(sampler.kernel, new_sample, log_likelihood, new_temp, old_state)
    new_weights = add_logdensity(old_state.log_weights, incr_weights)
    new_evidence = old_state.log_evidence + logsumexp(new_weights)
    normalized_weights = normalize_log_weights(new_weights)
    new_state = SmcState(new_sample, normalized_weights, log_likelihood, new_evidence, new_temp)

    resampled = maybe_resample(rng, new_state, sampler.log_resample_threshold)
    resampled.sample, resampled
end

# SmcKernels, implement:
# proposal(kernel): Getter for the proposal
# forward(kernel, new_sample, old_state): forward kernel for the proposed sample
# incremental_weights(kernel, new_sample, new_likelihood, new_temp, old_state): calculate the unnormalized incremental weights

propose(kernel, previous_sample, n_particles) = propose(proposal(kernel), previous_sample, n_particles)

"""
    ForwardProposalKernel(proposal)
Use the transition probability of the forward proposal for the backward L-kernel.
This results in a weight increment similar to a Metropolis-Hastings acceptance ratio.
(Increasing the efficiency of Sequential Monte Carlo samplers through the use of approximately optimal L-kernels, Green 2022)
"""
struct ForwardProposalKernel{Q}
    proposal::Q
end

proposal(kernel::ForwardProposalKernel) = kernel.proposal
forward(kernel::ForwardProposalKernel, new_sample, old_sample) = new_sample

"""
    increment_weights(kernel, new_sample, new_likelihood, new_temp, old_state)
Calculate the unnormalized incremental log using a "forward proposal L-kernel" (Increasing the Efficiency of Sequential Monte Carlo Samplers..., Green 2022).
The weights are updated similarly to a Metropolis-Hastings acceptance ratio.
"""
function incremental_weights(kernel::ForwardProposalKernel, new_sample::Sample, new_likelihood, new_temp, old_state::SmcState)
    forward = transition_probability(kernel.proposal, new_sample, old_state.sample)
    backward = transition_probability(kernel.proposal, old_state.sample, new_sample)
    logprob(new_sample) .+ backward - logprob(old_state.sample) .- forward
end

struct MhKernel{R,Q}
    rng::R
    proposal::Q
end

proposal(kernel::MhKernel) = kernel.proposal
forward(kernel::MhKernel, new_sample, old_sample) = mh_kernel(kernel.rng, proposal(kernel), new_sample, old_sample)

"""
    increment_weights(kernel, new_sample, new_likelihood, new_temp, old_state)
Calculate the unnormalized incremental log using an MCMC Kernel (Sequential Monte Carlo Samplers, Del Moral 2006).
For a likelihood tempered target γᵢ = p(z|θ)ᵠp(θ) the incremental weight formula simplifies to γ₂/γ₁ = p(z|θ₁)^(ϕ₂ - ϕ₁) (Efficient Sequential Monte-Carlo Samplers for Bayesian Inference, Nguyen 2016)
"""
# TODO revert to log_likelihood and do not temper in PosteriorModel since there is almost no expected performance gain since only ϕ₀=0
incremental_weights(::MhKernel, new_sample::Sample, new_likelihood, new_temp, old_state::SmcState) = (new_temp - old_state.temperature) .* old_state.log_likelihood
# TODO why is ith so much better?
# incremental_weights(::MhKernel, new_sample::Sample, new_likelihood, new_temp, old_state::SmcState) = (new_temp - old_state.temperature) .* new_likelihood

"""
incremental_weights(::MhKernel, new_sample::Sample, new_temp, old_state::SmcState) = (new_temp - old_state.temperature) .* old_state.log_likelihood

# Resampling

"""
    maybe_resample(rng, state, log_threshold)
Resample the variables of the `state` with their respective log-weights & -probabilities if the log effective sample size is smaller than the `log_threshold`
"""
function maybe_resample(rng::AbstractRNG, state::SmcState, log_threshold)
    if effective_sample_size(state.log_weights) < log_threshold
        # Resample variables
        indices = systematic_resampling_indices(rng, state.log_weights)
        vars = map(x -> @view(x[.., indices]), variables(state.sample))
        log_probs = logprob(state.sample)[indices]
        re_sample = Sample(vars, log_probs)
        # Reset weights
        log_weights = fill(-log(length(log_probs)), length(log_probs))
        SmcState(re_sample, log_weights, state.log_likelihood, state.log_evidence, state.temperature)
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

# Other common functions

"""
normalize_log_weights(log_weights)
    Normalization of the weights in the log domain using the log-sum-exp trick.
"""
normalize_log_weights(log_weights) = log_weights .- logsumexp(log_weights)
