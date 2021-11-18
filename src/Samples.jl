# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using TransformVariables

"""
    Sample(θ, p, t)
Might have a constrained parameter Domain, e.g. θᵢ ∈ ℝ⁺.
Consists of the current raw state `θ::Vector{Float64}`, the probability `p` and a transformation rule `t`.
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
    state(s)
State of the parameters in the model domain.
"""
state(s::Sample) = transform(s.t, collect(s.θ))

"""
    log_probability(s)
Jacobian-corrected posterior log probability of the of this sample.
"""
function log_probability(s::Sample)
    _, ℓ_t = transform_and_logjac(s.t, s.θ)
    s.p + ℓ_t
end

"""
    +(s, θ)
Add a raw state `θ` to the raw state (unconstrained domain) of the sample `s`.
"""
Base.:+(s::Sample{T,U}, θ::NamedTuple{T,V}) where {T,U,V} = @set s.θ = s.θ + reduce(vcat, θ)

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
Base.:-(s::Sample{T,U}, θ::NamedTuple{T,V}) where {T,U,V} = @set s.θ = s.θ - reduce(vcat, θ)

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
Base.:-(a::Sample{T,U}, b::Sample{T,V}) where {T,U,V} = a + b.θ

s = Sample([1.0, 2.0, 3.0], 1.0, as((; a = as_unit_interval, b = as(Vector, asℝ₊, 2))))
st = state(s)
state(s + st)