# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    SequentialMonteCarlo
Sequential Monte Carlo with systematic resampling and likelihood tempering via p(θ|z) ∝ p(z|θ)ᶲ p(θ).
"""
struct SequentialMonteCarlo{K,S} <: AbstractMCMC.AbstractSampler
    kernel::K
    temp_schedule::S
    n_particles::Int64
    log_relative_ess_threshold::Float64
end

Base.show(io::IO, s::SequentialMonteCarlo) = print(io, "SequentialMonteCarlo: $(s.kernel)")


struct SmcState{S<:Sample,W<:AbstractVector}
    # Contains multiple values for the variables
    sample::S
    log_weights::W
    log_evidence::Float64
    temperature::Float64
    log_relative_ess::Float64   # log(ESS/n_samples)
end

logevidence(state::SmcState) = state.log_evidence

function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, sampler::SequentialMonteCarlo)
    # This is an importance sampling step
    # rand on PosteriorModel samples from prior in unconstrained domain
    sample = rand(model, sampler.n_particles)
    # tempering starts with ϕ₀=0
    sample = tempered_logdensity_sample(model, sample, 0.0)

    # ϕ₀=0 → importance distribution = target density → wᵢ=1, normalized:
    normalized_log_weights = fill(-log(sampler.n_particles), sampler.n_particles)
    # IS normalizing constant: 1/n * ∑ₙ wᵢ = n_particles / n_particles = 1 → log(1) = 0
    state = SmcState(sample, normalized_log_weights, 0.0, 0.0, log_relative_ess(normalized_log_weights))

    state.sample, state
end

"""
    step(rng, model, sampler, state)
Generic SMC sampler according to 3.1.1. (Sequential Monte Carol Samplers, Del Moral 2006)
"""
function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, sampler::SequentialMonteCarlo, old_state::SmcState)
    # Schedule the likelihood tempering
    new_temp = increment_temperature(sampler.temp_schedule, old_state.temperature)

    # Draw new particles using the forward kernel
    proposed_sample = propose(sampler.kernel, old_state, sampler.n_particles)
    proposed_sample = tempered_logdensity_sample(model, proposed_sample, new_temp)
    new_sample = forward(sampler.kernel, proposed_sample, old_state.sample)

    # Update weights using backward kernel
    incr_weights = incremental_weights(sampler.kernel, new_sample, new_temp, old_state)
    new_weights = add_logdensity(old_state.log_weights, incr_weights)
    # Unnormalized new weights from (12) are the elements of (14) in the SMC paper
    new_evidence = old_state.log_evidence + logsumexp(new_weights)
    normalized_weights = normalize_log_weights(new_weights)
    new_state = SmcState(new_sample, normalized_weights, new_evidence, new_temp, log_relative_ess(normalized_weights))

    resampled = maybe_resample(rng, new_state, sampler.log_relative_ess_threshold)
    resampled.sample, resampled
end

# SmcKernels, must have a `proposal` field. 
# propose(kernel, old_state, n_particles): propose a new `Sample` using the old `SmcState`
# forward(kernel, new_sample, old_sample): forward kernel for the proposed sample
# incremental_weights(kernel, new_sample, new_temp, old_state::SmcState): calculate the unnormalized incremental weights

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
    increment_weights(kernel, new_sample, new_temp, old_state)
Calculate the unnormalized incremental log using a "forward proposal L-kernel" (Increasing the Efficiency of Sequential Monte Carlo Samplers..., Green 2022).
The weights are updated similarly to a Metropolis-Hastings acceptance ratio.
"""
function incremental_weights(kernel::ForwardProposalKernel, new_sample::Sample, new_temp, old_state::SmcState)
    forward = transition_probability(kernel.proposal, new_sample, old_state.sample)
    backward = transition_probability(kernel.proposal, old_state.sample, new_sample)
    logprobability(new_sample) .+ backward .- logprobability(old_state.sample) .- forward
end

struct MhKernel{R,Q}
    rng::R
    proposal::Q
end

Base.show(io::IO, k::MhKernel) = print(io, "MhKernel, $(k.proposal)")

propose(kernel::MhKernel, old_state, n_particles) = propose(kernel.proposal, old_state.sample, n_particles)
forward(kernel::MhKernel, new_sample, old_sample) = mh_kernel(kernel.rng, kernel.proposal, new_sample, old_sample)

"""
    increment_weights(kernel, new_sample, new_temp, old_state)
Calculate the unnormalized incremental log using an MCMC Kernel (Sequential Monte Carlo Samplers, Del Moral 2006).
For a likelihood tempered target γᵢ = p(z|θ)ᵠp(θ) the incremental weight formula simplifies to γ₂/γ₁ = p(z|θ₁)^(ϕ₂ - ϕ₁) (Efficient Sequential Monte-Carlo Samplers for Bayesian Inference, Nguyen 2016)
"""
incremental_weights(::MhKernel, new_sample::Sample, new_temp, old_state::SmcState) = (new_temp - old_state.temperature) .* loglikelihood(old_state.sample)

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
    increment_weights(kernel, new_sample, new_temp, old_state)
Bootstrap particle filter: tempered likelihood is the weight increment.
"""
incremental_weights(kernel::BootstrapKernel, new_sample::Sample, new_temp, old_state::SmcState) = loglikelihood(new_sample)

"""
    AdaptiveKernel(kernel)
Wraps an SMC kernel which is expected to use a symmetric proposal model, accessible via kernel.proposal. 
"""
struct AdaptiveKernel{R,K}
    rng::R
    kernel::K
end
# TODO I think it should be possible to implement an AdaptiveProposal and not do these hacks

function propose(kernel::AdaptiveKernel, old_state::SmcState, n_particles)
    internal = kernel.kernel
    # immutable does not modify the parameter only locally
    # TODO Should I use unbiased estimate compared to nguyenEfficientSequentialMonteCarlo2016 
    @reset internal.proposal = adaptive_mvnormal(kernel.rng, internal.proposal, old_state; corrected=true)
    propose(internal, old_state, n_particles)
end

forward(kernel::AdaptiveKernel, new_sample, old_sample) = forward(kernel.kernel, new_sample, old_sample)

incremental_weights(kernel::AdaptiveKernel, new_sample::Sample, new_temp, old_state::SmcState) = incremental_weights(kernel.kernel, new_sample, new_temp, old_state)

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
    # TODO assumes that the variables are grouped in vectors, e.g. [x,y,z] components of Translation. Otherwise we would have to assemble one large covariance matrix for the different variables and somehow tell the proposal how to divide it back into variables. Fallback could be to evaluate variance instead?
    Σ_vars = map(vars) do x
        # eltype required since weights would change it to Float64
        if x isa AbstractMatrix
            cov(Array(x), weights, 2; corrected=corrected) .|> quat_eltype(x)
        elseif x isa AbstractVector
            var(Array(x), weights; corrected=corrected) |> eltype(x)
        else
            # not positive definite
            0
        end
    end

    # Replace model with MvNormal moves
    nodes = map(names) do name
        # peaked likelihood - essentially one particle / sample with all the weight
        # Σ might be close to zero → Cholesky factorization fails
        Σ = Σ_vars[name]
        if isposdef(Σ)
            if Σ isa AbstractMatrix
                SimpleNode(name, rng, MvNormal, Σ)
            else
                SimpleNode(name, rng, KernelNormal, 0, Σ)
            end
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
    maybe_resample(rng, state, relative_ess_threshold)
Resample the variables of the `state` with their respective log-weights & -probabilities if the relative log effective sample size is smaller than the `log_relative_ess_threshold`
"""
maybe_resample(rng::AbstractRNG, state::SmcState, log_relative_ess_threshold) = state.log_relative_ess < log_relative_ess_threshold ? resample_systematic(rng, state) : state

"""
    resample_systematic(rng, state)
Systematic resampling scheme according to the log-weights.
Returns the resampled SmcState where all log-weights are equal.
"""
function resample_systematic(rng::AbstractRNG, state::SmcState)
    # Resample variables
    indices = systematic_resampling_indices(rng, state.log_weights)
    # TODO resampling all variables includin o & μ is expensive - would have reduce prior
    vars = map(variables(state.sample)) do x
        @view x[.., indices]
    end
    log_probs = logprobability(state.sample)[indices]
    log_likes = loglikelihood(state.sample)[indices]
    re_sample = Sample(vars, log_probs, log_likes)
    # Reset weights
    log_weights = fill(-log(length(log_probs)), length(log_probs))
    SmcState(re_sample, log_weights, state.log_evidence, state.temperature, state.log_relative_ess)
end

"""
    log_relative_ess(log_weights)
Expects normalized log_weights and calculates log(ESS)=log(1/∑(wᵢ²))=-log(∑exp(2*log(wᵢ)))-log(n_particles)
"""
log_relative_ess(log_weights) = -logsumexp(2 .* log_weights) - log(length(log_weights))

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
function normalize_log_weights(log_weights)
    # avoid division by zero / NaN in case all elements are -Inf
    denom = logsumexp(log_weights)
    isinf(denom) ? log_weights : (log_weights .- denom)
end
