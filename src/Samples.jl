# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using TransformVariables

"""
  AbstractSample
Abstract type of a sample which consists of its state `θ::NamedTuple{T, U}}` and the log_likelihood `ℓ::Float64`.
"""
abstract type AbstractSample{T,U} end

"""
  state(s)
State of the parameters in the model domain.
"""
state(s::AbstractSample) = s.θ

"""
    log_likelihood(s)
Likelihood of the state of this sample.
"""
log_likelihood(s::AbstractSample) = s.ℓ

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
    ConstrainedSample(θ, log_likelihood(s), t)
end

"""
    ConstrainedSample(s, t)
Conversion from ConstrainedSample to simple Sample.
"""
function Sample(s::ConstrainedSample)
    Sample(state(s), log_likelihood(s))
end

"""
    state(s)
State of the parameters in the model domain.
"""
state(s::ConstrainedSample) = transform(s.t, collect(s.θ))

"""
    log_likelihood(s)
Jacobian-corrected likelihood of the state of this sample.
"""
function log_likelihood(s::ConstrainedSample)
    _, ℓ_t = transform_and_logjac(s.t, collect(s.θ))
    s.ℓ + ℓ_t
end

"""
    merge(s, others...)
Combine factorized weighted samples (p(w1,w2)=p(w1)p(w2)).
Effectively this means, that the log-weights are summed up and the states are merged to one NamedTuple.
Note that the constraints are not part of this function.
"""
function Base.merge(s::AbstractSample{T}, others::AbstractSample{T}...) where {T}
    θ = merge(state(s), (state(x) for x in others)...)
    ℓ = log_likelihood(s) + sum([log_likelihood(x) for x in others])
    Sample(θ, ℓ)
end

# TODO only add same Types?
#TODO This would imply the conversion of a Sample to an ConstrainedSample, which I think I like
function add(a::V, b::W) where {T,U,V<:AbstractSample{T,U},W<:AbstractSample{T,U}}
    c = @set a.θ = NamedTuple{T,U}(collect(a.θ) + collect(b.θ))
    @set a.ℓ = -Inf
end

# Base .- (a::NamedTuple{T,U}, b::NamedTuple{T,U}) where {T,U} = NamedTuple{T,U}(collect(a) - collect(b))

# TODO remove
using Soss, MeasureTheory

m = @model begin
    e ~ Exponential(1)
    x ~ Exponential(e)
end

a = rand(m(e = 1))
ℓ(x) = logdensity(m | x)
tr = xform(m(;))
s = Sample(a, ℓ(a))
cs = ConstrainedSample(s, tr)
add(s, cs)
# transform_logdensity(tr, ℓ, s_inv - [10])
