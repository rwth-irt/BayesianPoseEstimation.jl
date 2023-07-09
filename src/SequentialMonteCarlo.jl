# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

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

Base.show(io::IO, s::SequentialMonteCarlo) = print(io, "SequentialMonteCarlo: $(s.kernel)")


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
    proposed_sample = propose(sampler.kernel, old_state, sampler.n_particles)
    log_prior, log_likelihood = prior_and_likelihood(model, proposed_sample)
    log_full = tempered_logdensity(log_prior, log_likelihood, new_temp)
    proposed_sample = set_logp(proposed_sample, log_full)
    new_sample = forward(sampler.kernel, proposed_sample, old_state.sample)

    # Update weights using backward kernel
    incr_weights = incremental_weights(sampler.kernel, new_sample, log_likelihood, new_temp, old_state)
    new_weights = add_logdensity(old_state.log_weights, incr_weights)
    # Unnormalized new weights from (12) are the elements of (14) in the SMC paper
    new_evidence = old_state.log_evidence + logsumexp(new_weights)
    normalized_weights = normalize_log_weights(new_weights)
    new_state = SmcState(new_sample, normalized_weights, log_likelihood, new_evidence, new_temp)

    resampled = maybe_resample(rng, new_state, sampler.log_resample_threshold)
    resampled.sample, resampled
end

# SmcKernels, must have a `proposal` field. 
# propose(kernel, old_state, n_particles): propose a new `Sample` using the old `SmcState`
# forward(kernel, new_sample, old_sample): forward kernel for the proposed sample
# incremental_weights(kernel, new_sample, new_likelihood, new_temp, old_state::SmcState): calculate the unnormalized incremental weights

"""
    ForwardProposalKernel(proposal)
Use the transition probability of the forward proposal for the backward L-kernel.
This results in a weight increment similar to a Metropolis-Hastings acceptance ratio.
(Increasing the efficiency of Sequential Monte Carlo samplers through the use of approximately optimal L-kernels, Green 2022)
"""
struct ForwardProposalKernel{Q}
    proposal::Q
end

Base.show(io::IO, k::ForwardProposalKernel) = print(io, "ForwardProposalKernel, $(k.proposal)")

propose(kernel::ForwardProposalKernel, old_state, n_particles) = propose(kernel.proposal, old_state.sample, n_particles)
forward(kernel::ForwardProposalKernel, new_sample, old_sample) = new_sample

"""
    increment_weights(kernel, new_sample, new_likelihood, new_temp, old_state)
Calculate the unnormalized incremental log using a "forward proposal L-kernel" (Increasing the Efficiency of Sequential Monte Carlo Samplers..., Green 2022).
The weights are updated similarly to a Metropolis-Hastings acceptance ratio.
"""
function incremental_weights(kernel::ForwardProposalKernel, new_sample::Sample, new_likelihood, new_temp, old_state::SmcState)
    forward = transition_probability(kernel.proposal, new_sample, old_state.sample)
    backward = transition_probability(kernel.proposal, old_state.sample, new_sample)
    logprob(new_sample) .+ backward .- logprob(old_state.sample) .- forward
end

struct MhKernel{R,Q}
    rng::R
    proposal::Q
end

Base.show(io::IO, k::MhKernel) = print(io, "MhKernel, $(k.proposal)")

propose(kernel::MhKernel, old_state, n_particles) = propose(kernel.proposal, old_state.sample, n_particles)
forward(kernel::MhKernel, new_sample, old_sample) = mh_kernel(kernel.rng, kernel.proposal, new_sample, old_sample)

"""
    increment_weights(kernel, new_sample, new_likelihood, new_temp, old_state)
Calculate the unnormalized incremental log using an MCMC Kernel (Sequential Monte Carlo Samplers, Del Moral 2006).
For a likelihood tempered target γᵢ = p(z|θ)ᵠp(θ) the incremental weight formula simplifies to γ₂/γ₁ = p(z|θ₁)^(ϕ₂ - ϕ₁) (Efficient Sequential Monte-Carlo Samplers for Bayesian Inference, Nguyen 2016)
"""
incremental_weights(::MhKernel, new_sample::Sample, new_likelihood, new_temp, old_state::SmcState) = (new_temp - old_state.temperature) .* old_state.log_likelihood

"""
    BootstrapKernel(proposal)
Kernel which results in a bootstrap SIR particle filter.
Uses the transition prior probability as importance function and the weight increment is the likelihood.
(An invitation to sequential Monte Carlo samplers, Dai 2022)
"""
struct BootstrapKernel{Q}
    proposal::Q
end

Base.show(io::IO, k::BootstrapKernel) = print(io, "BootstrapKernel, $(k.proposal)")

propose(kernel::BootstrapKernel, old_state, n_particles) = propose(kernel.proposal, old_state.sample, n_particles)
forward(kernel::BootstrapKernel, new_sample, old_sample) = new_sample

"""
    increment_weights(kernel, new_sample, tempered_likelihood, new_temp, old_state)
Bootstrap particle filter: tempered likelihood is the weight increment.
"""
incremental_weights(kernel::BootstrapKernel, new_sample::Sample, new_likelihood, new_temp, old_state::SmcState) = new_likelihood

"""
    AdaptiveKernel(kernel)
Wraps an SMC kernel which is expected to use a symmetric proposal model, accessible via kernel.proposal. 
"""
struct AdaptiveKernel{R,K}
    rng::R
    kernel::K
end
# TODO Should I enforce symmetric proposal model? How?

# TODO change all propose functions for SmcKernels to SmcState
function propose(kernel::AdaptiveKernel, old_state::SmcState, n_particles)
    internal = kernel.kernel
    # TODO Fragile? Unit test?
    # NOTE immutable does not modify the parameter only locally
    # TODO Should I use unbiased estimate compared to nguyenEfficientSequentialMonteCarlo2016 
    @reset internal.proposal = adaptive_mvnormal(kernel.rng, internal.proposal, old_state; corrected=false)
    propose(internal, old_state, n_particles)
end

forward(kernel::AdaptiveKernel, new_sample, old_sample) = forward(kernel.kernel, new_sample, old_sample)

incremental_weights(kernel::AdaptiveKernel, new_sample::Sample, new_likelihood, new_temp, old_state::SmcState) = incremental_weights(kernel.kernel, new_sample, new_likelihood, new_temp, old_state)

# TEST
"""
    adaptive_mvnormal(proposal::Proposal, state::SmcState; [corrected=true])
Replaces the model of the proposal with multivariate normal distributions.
These distributions are zero-centered and have the covariance of the `state`'s distribution.

If the state has close to zero covariance, the Cholesky factorization fails and the original proposal distribution is returned for that variable.
"""
function adaptive_mvnormal(rng::AbstractRNG, proposal::Proposal{names}, state::SmcState; corrected=true) where {names}
    vars = variables(state.sample)[names]
    # analytic / reliability weights describe an importance of each observation
    weights = state.log_weights .|> exp |> AnalyticWeights
    Σ_vars = map(vars) do x
        # TODO Array(x) because cov is not implemented for views...
        cov(Array(x), weights, 2; corrected=corrected) .|> quat_eltype(x)
    end

    # Replace model with MvNormal moves
    nodes = map(names) do name
        # peaked likelihood - essentially one particle / sample with all the weight
        # Σ might be close to zero → Cholesky factorization fails
        Σ = Σ_vars[name]
        if isposdef(Σ)
            SimpleNode(name, rng, MvNormal, Σ_vars[name])
        else
            # Fall back to original proposal distribution
            proposal.model[name]
        end
    end
    @set proposal.model = NamedTuple{names}(nodes)
end

# Weights alter the precision of the covariance matrices
quat_eltype(::AbstractArray{Quaternion{T}}) where {T} = T
quat_eltype(::AbstractArray{T}) where {T} = T

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
