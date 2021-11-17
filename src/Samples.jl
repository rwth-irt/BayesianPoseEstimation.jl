# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using TransformVariables

"""
  AbstractSample
Abstract type of a sample which consists of its state `θ::NamedTuple{T, U}}` and the posterior log probability `ℓ::Float64`.
Convention: The raw state values must be in the unconstrained domain.
"""
abstract type AbstractSample{T,U} end

"""
  state(s)
State of the parameters in the model domain.
"""
state(s::AbstractSample) = s.θ

"""
    log_probability(s)
Posterior log probability of the of this sample.
"""
log_probability(s::AbstractSample) = s.ℓ

"""
    is_constrained(::AbstractSample)
Implement IsConstrained trait.
"""
is_constrained(::AbstractSample) = IsConstrained{false}()

"""
    Sample
Consists of the current state `θ` and the likelihood of the state `ℓ`.
"""
struct Sample{T,U} <: AbstractSample{T,U}
    θ::NamedTuple{T,U}
    ℓ::Float64
end

"""
    ConstrainedSample
Has a constrained parameter Domain, e.g. θᵢ ∈ ℝ⁺.
Consists of the current **transformed** state `θ`, the probability of having transitioned to this state `q` and the likelihood of the state `ℓ`.
"""
struct ConstrainedSample{T,U,V} <: AbstractSample{T,U}
    θ::NamedTuple{T,U}
    ℓ::Float64
    t::TransformVariables.TransformTuple{NamedTuple{T,V}}
end

"""
    ConstrainedSample(s, t)
Constructor to create a ConstrainedSample from an unconstrained sample `s`.
`t` is a transformation from the unconstrained to the constrained space.
"""
function ConstrainedSample(s::AbstractSample{T}, t::TransformVariables.TransformTuple) where {T}
    θ = NamedTuple{T}(inverse(t, state(s)))
    ConstrainedSample(θ, log_probability(s), t)
end

"""
    state(s)
State of the parameters in the model domain.
"""
state(s::ConstrainedSample) = transform(s.t, collect(s.θ))

"""
    log_probability(s)
Jacobian-corrected posterior log probability of the of this sample.
"""
function log_probability(s::ConstrainedSample)
    _, ℓ_t = transform_and_logjac(s.t, collect(s.θ))
    s.ℓ + ℓ_t
end

"""
    is_constrained(::ConstrainedSample)
Implement IsConstrained trait.
"""
is_constrained(::ConstrainedSample) = IsConstrained{true}()

"""
    ConstrainedSample(s, t)
Conversion from ConstrainedSample to simple Sample.
"""
function Sample(s::ConstrainedSample)
    Sample(state(s), log_probability(s))
end

Base.convert(::Type{Sample}, x) = Sample(x)

"""
    merge(s, others...)
Combine factorized weighted samples (p(w1,w2)=p(w1)p(w2)).
Effectively this means, that the log-weights are summed up and the states are merged to one NamedTuple.
Note that the constraints are not part of this function.
"""
function Base.merge(s::AbstractSample, others::AbstractSample...) where {T}
    θ = merge(state(s), (state(x) for x in others)...)
    ℓ = log_probability(s) + sum([log_probability(x) for x in others])
    Sample(θ, ℓ)
end

"""
    +(s, θ)
Add a raw state `θ` to the raw state (unconstrained domain) of the sample `s`.
"""
function Base.:+(s::AbstractSample{T,U}, θ::NamedTuple{T,U}) where {T,U}
    c = @set s.θ = NamedTuple{T,U}(collect(s.θ) + collect(θ))
    @set c.ℓ = -Inf
end

"""
    +(a, b)
Add the raw states (unconstrained domain) of two samples.
Only same type is supported to prevent surprises in the return type.
"""
Base.:+(a::V, b::V) where {T,U,V<:AbstractSample{T,U}} = a + b.θ

"""
    -(s, θ)
Subtract a raw state `θ` from the raw state (unconstrained domain) of the sample `s`.
"""
function Base.:-(s::AbstractSample{T,U}, θ::NamedTuple{T,U}) where {T,U}
    c = @set s.θ = NamedTuple{T,U}(collect(s.θ) - collect(θ))
    @set c.ℓ = -Inf
end

"""
    -(a, b)
Subtract the raw states (unconstrained domain) of two samples.
Only same type is supported to prevent surprises in the return type.
"""
Base.:-(a::V, b::V) where {T,U,V<:AbstractSample{T,U}} = a - b.θ
