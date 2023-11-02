# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 


"""
    MetropolisHastings
Different MetropolisHastings samplers only differ by their proposals.   
"""
struct MetropolisHastings{Q,S} <: AbstractMCMC.AbstractSampler
    proposal::Q
    temp_schedule::S
end

Base.show(io::IO, ::MetropolisHastings) = print(io, "MetropolisHastings")

struct MCMCState{S<:Sample}
    sample::S
    temperature::Float64
end

"""
    step(rng, model, sampler)
Implementing the AbstractMCMC interface for the initial step.
Proposes one sample from the prior of the sampler.
"""
function AbstractMCMC.step(::AbstractRNG, model::PosteriorModel, sampler::MetropolisHastings)
    # rand on PosteriorModel samples from prior in unconstrained domain
    sample = rand(model)
    # initial evaluation of the posterior logdensity
    sample = tempered_logdensity_sample(model, sample, 0.0)
    # sample, state are the same for MH
    sample, MCMCState(sample, 0.0)
end

"""
    step(rng, model, sampler, state)
Implementing the AbstractMCMC interface for steps given a state from the last step.
"""
function AbstractMCMC.step(rng::AbstractRNG, model::PosteriorModel, sampler::MetropolisHastings, old_state::MCMCState)
    # Schedule the likelihood tempering
    new_temp = increment_temperature(sampler.temp_schedule, old_state.temperature)
    # Draw new samples using MH kernel
    proposed = propose(sampler.proposal, old_state.sample)
    proposed = tempered_logdensity_sample(model, proposed, new_temp)
    result = mh_kernel(rng, sampler.proposal, proposed, old_state.sample)
    # sample, state
    result, MCMCState(result, new_temp)
end

"""
    mh_kernel(rng, proposal, proposed, previous)
Metropolis-Hastings transition kernel for the `proposed` and `previous` sample.
For both samples, `logprob(sample)` must be valid.
Vectorization for multiple proposals is supported out of box.
"""
function mh_kernel(rng, proposal, proposed, previous)
    α = acceptance_ratio(proposal, proposed, previous)
    # Easier to calculate and more natural to mutate the proposed instead of the previous sample
    rejected = should_reject(rng, α)
    reject_barrier(rejected, proposed, previous)
end

"""
    acceptance_ratio(proposal, proposed, previous)
Vectorized Metropolis-Hastings acceptance ratio via broadcasting.
"""
acceptance_ratio(proposal::Proposal, proposed::Sample, previous::Sample) = (
    logprobability(proposed) .-
    logprobability(previous) .+
    transition_probability(proposal, previous, proposed) .-
    transition_probability(proposal, proposed, previous)
)

"""
    should_reject(rng, log_α)
Returns an (Array of) Bool which is true, if the element should be rejected according to the logarithmic acceptance ratio `log_α`.
Calculation of rejection is easier since for the random number u: log([0,1]) ∈ [-∞,0].
Thus, `should_reject` always returns `false`, if α >= 1 ⇔ log(α) >= 0.
"""
should_reject(rng::AbstractRNG, log_α::AbstractArray) = log.(rand(rng, length(log_α))) .> log_α
# Avoid generating a single element vector
should_reject(rng::AbstractRNG, log_α::Real) = log(rand(rng)) > log_α

"""
    reject_barrier(reject)
Type stable implementation for single and multiple proposals.
"""
function reject_barrier(rejected::AbstractArray{Bool}, proposed, previous)
    vars = map_intersect(variables(proposed), variables(previous)) do prop, prev
        # WARN copying both to avoid weird illegal access errors if previous<:SubArray{<:Any,<:Any,<:CuArray}
        reject_vectorized!(rejected, copy(prop), copy(prev))
    end
    log_prob = reject_vectorized!(rejected, copy(logprobability(proposed)), logprobability(previous))
    log_like = reject_vectorized!(rejected, copy(loglikelihood(proposed)), loglikelihood(previous))
    # No mutation in scalar case...
    Sample(vars, log_prob, log_like)
end
# Scalar case
reject_barrier(rejected::Bool, proposed, previous) = rejected ? previous : proposed
# DeterminsticNode as root
reject_barrier(rejected::AbstractVector{Bool}, proposed, previous) = previous

"""
    reject_vectorized!(rejected, proposed, previous)
Modifies the proposed array, by replacing the values from the previous array where rejected is true.
The selection is done along the last dimension of the arrays.
"""
function reject_vectorized!(rejected::AbstractVector{Bool}, proposed::AbstractArray, previous::AbstractArray)
    if ndims(proposed) == ndims(previous)
        @views proposed[.., rejected] .= previous[.., rejected]
    elseif ndims(proposed) > ndims(previous)
        @view(proposed[.., rejected]) .= previous
    else
        throw(ArgumentError("Rejecting ndims(proposed)=$(ndims(proposed)) < ndims(previous)=$(ndims(previous)) not possible"))
    end
    proposed
end
# Previous as scalar
function reject_vectorized!(rejected::AbstractVector{Bool}, proposed::AbstractArray, previous::Real)
    @view(proposed[.., rejected]) .= previous
    proposed
end

# Move rejected vector to the GPU
reject_vectorized!(rejected::Vector{Bool}, proposed::CuArray, previous::AbstractArray) = reject_vectorized!(CuArray(rejected), proposed, previous)
reject_vectorized!(rejected::Vector{Bool}, proposed::CuArray, previous::Real) = reject_vectorized!(CuArray(rejected), proposed, previous)
