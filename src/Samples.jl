# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# bundle_samples for the Sample type
using AbstractMCMC, TupleVectors
using Accessors
using MeasureTheory, Soss
using Random
using TransformVariables

"""
    Sample(θ, p, t)
Might have a constrained parameter Domain, e.g. θᵢ ∈ ℝ⁺.
Consists of the current raw state `θ::Vector{Float64}`, the (uncorrected) posterior probability `p` and a transformation rule `t`.
Samples are generically typed by `T` for the variable names and `U` to specify their respective domain transformation.
"""
struct Sample{T,U}
    θ::Vector{Float64}
    p::Float64
    t::TransformVariables.TransformTuple{NamedTuple{T,U}}
end

"""
    Sample(θ, p, t)
Create a sample from `θ::NamedTuple` containing the current state and its probability `p`.
`t` is a transformation from the unconstrained to the constrained space.
"""
function Sample(θ::NamedTuple, p, t::TransformVariables.TransformTuple)
    θ_raw = inverse(t, θ)
    Sample(θ_raw, p, t)
end

"""
    Sample(m)
Create a sample by sampling from a model.
"""
function Sample(rng::AbstractRNG, m::Soss.AbstractModel)
    θ = rand(rng, m)
    # xform requires ConditionalModel, conditioning on nothing does not change the model except converting it a ConditionalModel
    tr = xform(m | (;))
    Sample(θ, -Inf, tr)
end

"""
    Sample(m)
Create a sample by sampling from a model.
"""
Sample(m::Soss.AbstractModel) = Sample(Random.GLOBAL_RNG, m)

"""
    state(s)
State of the parameters in the model domain.
Returns a `NamedTuple` with the variable names and their model domain values.
"""
state(s::Sample) = transform(s.t, s.θ)

"""
    raw_state(s)
State of the parameters in the unconstrained domain.
Returns a `NamedTuple` with the variable names and their model domain values.
"""
raw_state(s::Sample{T,U}) where {T,U} = NamedTuple{T}(s.θ)

"""
    log_probability(s)
Jacobian-corrected posterior log probability of the sample.
"""
function log_probability(s::Sample)
    _, ℓ_t = transform_and_logjac(s.t, s.θ)
    s.p + ℓ_t
end

"""
    logdensity(m, s)
Non-corrected logdensity of the of the sample `s` given the measure `m`.
"""
MeasureTheory.logdensity(m::AbstractMeasure, s::Sample) = logdensity(m, state(s))

"""
    logdensity(m, s)
Non-corrected logdensity of the of the sample `s` given the model `m`.
"""
# Required to solve ambiguity for ConditionalModel
MeasureTheory.logdensity(m::Soss.ConditionalModel, s::Sample) = logdensity(m, state(s))

"""
    flatten(x)
Flattens x to return a 1D array.
"""
flatten(x) = collect(Iterators.flatten(x))

"""
    +(s, θ)
Add a raw state `θ` to the raw state (unconstrained domain) of the sample `s`.
"""
Base.:+(s::Sample{T,U}, θ::NamedTuple{T,V}) where {T,U,V} = @set s.θ = s.θ + flatten(θ)

"""
    +(s, θ)
Add a raw state `θ` to the raw state (unconstrained domain) of the sample `s`.
"""
Base.:+(s::Sample{T,U}, θ::AbstractVector) where {T,U,V} = @set s.θ = s.θ + θ

"""
    +(a, b)
Add the raw states (unconstrained domain) of two samples.
Only same type is supported to prevent surprises in the return type.
"""
Base.:+(a::Sample{T,U}, b::Sample{T,V}) where {T,U,V} = a + b.θ

"""
    -(s, θ)
Subtract raw state `θ` from the raw state (unconstrained domain) of the sample `s`.
"""
Base.:-(s::Sample{T,U}, θ::NamedTuple{T,V}) where {T,U,V} = @set s.θ = s.θ - flatten(θ)

"""
    -(s, θ)
Subtract raw state `θ` from the raw state (unconstrained domain) of the sample `s`.
"""
Base.:-(s::Sample{T,U}, θ::AbstractVector) where {T,U,V} = @set s.θ = s.θ - θ

"""
    +(a, b)
Subtract the raw states (unconstrained domain) of two samples.
Only same type is supported to prevent surprises in the return type.
"""
Base.:-(a::Sample{T,U}, b::Sample{T,V}) where {T,U,V} = a - b.θ

"""
    merge(as, bs...)
Left-to-Right merges the samples as with bs.
This means the the rightmost state is used.
The original transformation rule of as is kept to ensure compatibility with the original sample.
Merging the log probabilities does not make sense without evaluating against the overall model, thus it is -Inf
"""
function Base.merge(as::Sample, bs::Sample...)
    merged_state = merge(state(as), map(x -> state(x), bs)...)
    Sample(merged_state, -Inf, as.t)
end

"""
    bundle_samples(samples, model, sampler, state, chain_type[; kwargs...])
Bundle all `samples` that were sampled from the `model` with the given `sampler` in a chain.
The final `state` of the `sampler` can be included in the chain. The type of the chain can
be specified with the `chain_type` argument.
By default, this method returns `samples`.
"""
function AbstractMCMC.bundle_samples(
    samples::Vector{<:Sample},
    ::AbstractMCMC.AbstractModel,
    ::AbstractMCMC.AbstractSampler,
    ::Any,
    ::Type{TupleVector};
    start = 1,
    step = 1
)
    states = map(state, samples)
    TupleVector(states[start:step:end])
end