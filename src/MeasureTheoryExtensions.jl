# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using LogExpFunctions
using MeasureTheory
using Random
using StatsBase
using TransformVariables

"""
    UniformInterval(a, b)
The *continuous uniform distribution* over an interval ``[a, b]`` has probability density function
```math
f(x; a, b) = \\frac{1}{b - a}, \\quad a \\le x \\le b
```
"""

@parameterized UniformInterval(a, b) ≪ Lebesgue(ℝ)

function MeasureTheory.basemeasure(d::UniformInterval{(:a, :b)})
    inbounds(x) = d.a <= x <= d.b
    constℓ = 0.0
    varℓ() = -log(d.b - d.a)
    base = Lebesgue(ℝ)
    FactoredBase(inbounds, constℓ, varℓ, base)
end

MeasureTheory.distproxy(d::UniformInterval{(:a, :b)}) = Dists.Uniform(d.a, d.b)

MeasureTheory.logdensity(d::UniformInterval{(:a, :b)}, x) = 0.0

TransformVariables.as(d::UniformInterval{(:a, :b)}) = as(Real, d.a, d.b)

"""
    CircularUniform
Similar to the UniformInterval distribution but fixed to the interval [0,2π].
Uses the CircularTransform for continuity.
```
"""

@parameterized CircularUniform() ≪ Lebesgue(ℝ)

function MeasureTheory.basemeasure(::CircularUniform{()})
    inbounds(x) = 0 <= x <= 2π
    constℓ = -log(2π)
    varℓ() = 0.0
    base = Lebesgue(ℝ)
    FactoredBase(inbounds, constℓ, varℓ, base)
end

MeasureTheory.distproxy(::CircularUniform{()}) = Dists.Uniform(0, 2π)

MeasureTheory.logdensity(::CircularUniform{()}, x) = -log(2π)

TransformVariables.as(::CircularUniform{()}) = as○


"""
    MixtureMeasure
Mixture of several Measures `components` which are associated with their corresponding `weights`.
p(θ) = w₁p(θ₁) + w₂p(θ₂) + ...
"""
struct MixtureMeasure{T<:Tuple} <: AbstractMeasure
    components::T # Tuple instead of array for type safety with different types
    log_weights::Vector{Float64}
    function MixtureMeasure(components, weights::Vector{Float64})
        components_tuple = Tuple(components)
        new{typeof(components_tuple)}(Tuple(components), log.(weights / sum(weights)))
    end
end

MixtureMeasure(components, weights::Vector{T}) where {T<:Number} = MixtureMeasure(components, Float64.(weights))

Base.show(io::IO, d::MixtureMeasure) = print(io, "MixtureMeasure\ncomponents: [$(d.components)]\nlog weights: $(d.log_weights))")

MeasureTheory.logdensity(μ::MixtureMeasure, x) = logsumexp(log_w + MeasureTheory.logdensity(m, x) for (log_w, m) in zip(μ.log_weights, μ.components))

MeasureTheory.basemeasure(::MixtureMeasure) = Lebesgue(ℝ)

StatsBase.sample(rng::AbstractRNG, d::MixtureMeasure) = d.components[sample(rng, Weights(exp.(d.log_weights)))]
Base.rand(rng::AbstractRNG, T::Type, μ::MixtureMeasure) = rand(rng, T, sample(rng, μ))

"""
    BinaryMixture
Mixture of several Measures `components` which are associated with their corresponding `weights`.
p(θ) = w₁p(θ₁) + w₂p(θ₂) + ...
"""
struct BinaryMixture{T,U} <: AbstractMeasure
    c1::T
    c2::U
    log_w1::Float64
    log_w2::Float64
    BinaryMixture(c1::T, c2::U, w1::Float64, w2::Float64) where {T,U} = new{T,U}(c1, c2, log(w1 / (w1 + w2)), log(w2 / (w1 + w2)))
end

BinaryMixture(c1, c2, w1::Number, w2::Number) = BinaryMixture(c1, c2, Float64(w1), Float64(w2))

Base.show(io::IO, d::BinaryMixture) = print(io, "BinaryMixture\ncomponents: $(d.c1), $(d.c2) \nlog weights: $(d.log_w1), $(d.log_w2)")

MeasureTheory.logdensity(μ::BinaryMixture, x)::Float64 = logaddexp(μ.log_w1 + logdensity(μ.c1, x), μ.log_w2 + logdensity(μ.c2, x))

MeasureTheory.basemeasure(::BinaryMixture) = Lebesgue(ℝ)

function Base.rand(rng::AbstractRNG, T::Type, μ::BinaryMixture)
    if log(rand(rng)) < μ.log_w1
        rand(rng, T, μ.c1)
    else
        rand(rng, T, μ.c2)
    end
end
