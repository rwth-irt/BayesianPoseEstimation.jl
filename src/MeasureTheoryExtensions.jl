# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

using DensityInterface
using LogExpFunctions
using MeasureTheory
using KeywordCalls
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

@parameterized UniformInterval() ≪ Lebesgue(ℝ)
@kwstruct UniformInterval(a, b)
UniformInterval(a, b) = UniformInterval((a=a, b=b))

@inline MeasureTheory.insupport(d::UniformInterval{(:a, :b)}, x) = d.a ≤ x ≤ d.b

function MeasureTheory.basemeasure(d::UniformInterval{(:a, :b)})
    constℓ = 0.0
    varℓ() = -log(d.b - d.a)
    base = Lebesgue(ℝ)
    FactoredBase(constℓ, varℓ, base)
end

MeasureTheory.proxy(d::UniformInterval{(:a, :b)}) = Dists.Uniform(d.a, d.b)

Base.rand(rng::AbstractRNG, ::Type{T}, d::UniformInterval) where {T} = rand(rng, proxy(d))

MeasureTheory.density_def(d::UniformInterval{(:a, :b)}, x) = 1.0
MeasureTheory.logdensity_def(d::UniformInterval{(:a, :b)}, x) = 0.0

TransformVariables.as(d::UniformInterval{(:a, :b)}) = as(Real, d.a, d.b)


"""
    CircularUniform
Similar to the UniformInterval distribution but fixed to the interval [0,2π].
Uses the CircularTransform for continuity.
```
"""

@parameterized CircularUniform() ≪ Lebesgue(ℝ)
@kwstruct CircularUniform()
@inline MeasureTheory.insupport(::CircularUniform, x) = 0 ≤ x ≤ 2π

function MeasureTheory.basemeasure(::CircularUniform{()})
    constℓ = -log(2π)
    varℓ() = 0.0
    base = Lebesgue(ℝ)
    FactoredBase(constℓ, varℓ, base)
end

MeasureTheory.proxy(::CircularUniform{()}) = Dists.Uniform(0, 2π)

Base.rand(rng::AbstractRNG, ::Type{T}, d::CircularUniform) where {T} = rand(rng, proxy(d))

MeasureTheory.density_def(::CircularUniform{(:a, :b)}, x) = 1.0
MeasureTheory.logdensity_def(::CircularUniform{()}, x) = 0.0

TransformVariables.as(::CircularUniform{()}) = as○


"""
    MixtureMeasure
Mixture of several Measures `components` which are associated with their corresponding `weights`.
p(θ) = w₁p(θ₁) + w₂p(θ₂) + ...
"""
struct MixtureMeasure{T<:Tuple} <: AbstractMeasure
    components::T # Tuple instead of array for type safety with different types
    weights::Weights{Float64,Float64,Vector{Float64}}

    function MixtureMeasure(components::T, weights::Vector{<:Number}) where {T<:Tuple}
        w_sum = sum(weights)
        new{T}(components, Weights(weights / w_sum, w_sum))
    end
end

MeasureTheory.insupport(d::MixtureMeasure, x) = reduce(&, d.components)

Base.show(io::IO, d::MixtureMeasure) = print(io, "MixtureMeasure\n  components: [$(d.components)]\n  weights: $(d.weights))")

MeasureTheory.logdensity_def(μ::MixtureMeasure, x)::Float64 = logsumexp(log(w) + logdensityof(m, x) for (w, m) in zip(μ.weights, μ.components))
MeasureTheory.basemeasure(::MixtureMeasure) = Lebesgue(ℝ)

Base.rand(rng::AbstractRNG, T::Type, μ::MixtureMeasure) = rand(rng, T, μ.components[sample(rng, μ.weights)])

"""
    BinaryMixture
Mixture of two Measures as `components` which are associated with their corresponding `weights`.
p(θ) = w₁p(θ₁) + w₂p(θ₂)
"""
struct BinaryMixture{T,U} <: AbstractMeasure
    c1::T
    c2::U
    log_w1::Float64
    log_w2::Float64
    BinaryMixture(c1::T, c2::U, w1::Float64, w2::Float64) where {T,U} = new{T,U}(c1, c2, log(w1 / (w1 + w2)), log(w2 / (w1 + w2)))
end

BinaryMixture(c1, c2, w1::Number, w2::Number) = BinaryMixture(c1, c2, Float64(w1), Float64(w2))

MeasureTheory.insupport(d::BinaryMixture, x) = insupport(d.c1, x) && insupport(d.c2, x)

Base.show(io::IO, d::BinaryMixture) = print(io, "BinaryMixture\ncomponents: $(d.c1), $(d.c2) \n  log weights: $(d.log_w1), $(d.log_w2)")

MeasureTheory.logdensity_def(μ::BinaryMixture, x)::Float64 = logaddexp(μ.log_w1 + logdensityof(μ.c1, x), μ.log_w2 + logdensityof(μ.c2, x))
MeasureTheory.basemeasure(::BinaryMixture) = Lebesgue(ℝ)

function Base.rand(rng::AbstractRNG, T::Type, μ::BinaryMixture)
    if log(rand(rng)) < μ.log_w1
        rand(rng, T, μ.c1)
    else
        rand(rng, T, μ.c2)
    end
end

"""
    as(d::MvNormal)
Support xform for multivariate normal distributions
"""
TransformVariables.as(d::MvNormal) = d |> MeasureTheory.proxy |> as
