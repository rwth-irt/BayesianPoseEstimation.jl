# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using CUDA
using MCMCDepth
using MeasureTheory
using LogExpFunctions
using Random
using TransformVariables

# isbitstype types which correspond to MeasureTheory measures
"""
Interface: Implement `gpu_measure(cpu_measure)` & `cpu_measure(gpu_measure)`.
"""
abstract type AbstractGpuMeasure <: AbstractMeasure end
Random.rand!(d::AbstractGpuMeasure, M::CuArray) = rand!(CURAND.default_rng(), d, M)
Base.rand(rng::AbstractRNG, d::AbstractGpuMeasure, dims::Integer...) = rand!(rng, d, CuArray{Float32}(undef, dims))
Base.rand(d::AbstractGpuMeasure, dims::Integer...) = rand!(d, CuArray{Float32}(undef, dims))

# GpuNormal

struct GpuNormal <: AbstractGpuMeasure
    μ::Float32
    σ::Float32
end

GpuNormal(::Normal{()}) = GpuNormal(0.0, 1.0)
GpuNormal(d::Normal{(:μ, :σ)}) = GpuNormal(d.μ, d.σ)

gpu_measure(d::Normal) = GpuNormal(d)
cpu_measure(d::GpuNormal) = Normal(d.μ, d.σ)

Base.show(io::IO, d::GpuNormal) = print(io, "GpuNormal, μ: $(d.μ), σ: $(d.σ)")

function MeasureTheory.logdensity(d::GpuNormal, x::T) where {T}
    μ = d.μ
    σ² = d.σ^2
    -T(0.5) * ((x - μ)^2 / σ²)
end

MeasureTheory.logpdf(d::GpuNormal, x::T) where {T} = logdensity(d, x) - log(d.σ) - log(sqrt(T(2π)))

Random.rand!(curand_rng::AbstractRNG, d::GpuNormal, M::CuArray) = randn!(curand_rng, M; mean=d.μ, stddev=d.σ)

TransformVariables.as(::GpuNormal) = asℝ

# GpuExponential
struct GpuExponential <: AbstractGpuMeasure
    λ::Float32
end

GpuExponential(::Exponential{()}) = GpuExponential(1.0)
GpuExponential(d::Exponential{(:λ,)}) = GpuExponential(d.λ)
GpuExponential(d::Exponential{(:β,)}) = GpuExponential(1 / d.β)

gpu_measure(d::Exponential) = GpuExponential(d)
cpu_measure(d::GpuExponential) = Exponential{(:λ,)}(d.λ)

Base.show(io::IO, d::GpuExponential) = print(io, "GpuExponential, λ: $(d.λ)")

MeasureTheory.logdensity(d::GpuExponential, x) = -d.λ * x
MeasureTheory.logpdf(d::GpuExponential, x) = logdensity(d, x) + log(d.λ)

uniform_to_exp(x, λ) = log(x) / (-λ)

function Random.rand!(curand_rng::AbstractRNG, d::GpuExponential, M::CuArray)
    rand!(curand_rng, M)
    map!(M, M) do x
        uniform_to_exp(x, d.λ)
    end
end

TransformVariables.as(::GpuExponential) = asℝ₊

# GpuUniformInterval

struct GpuUniformInterval <: AbstractGpuMeasure
    a::Float32
    b::Float32
end

GpuUniformInterval(::UniformInterval{()}) = GpuUniformInterval(0.0, 1.0)
GpuUniformInterval(d::UniformInterval{(:a, :b)}) = GpuUniformInterval(d.a, d.b)

gpu_measure(d::UniformInterval) = GpuUniformInterval(d)
cpu_measure(d::GpuUniformInterval) = UniformInterval(d.a, d.b)

Base.show(io::IO, d::GpuUniformInterval) = print(io, "GpuUniformInterval, a: $(d.a), b: $(d.b)")

MeasureTheory.logdensity(d::GpuUniformInterval, x::T) where {T} = d.a <= x <= d.b ? zero(T) : -typemax(T)
MeasureTheory.logpdf(d::GpuUniformInterval, x) = logdensity(d, x) - log(d.b - d.a)

scale_uniform(x, a, b) = x * (b - a) + a

function Random.rand!(curand_rng::AbstractRNG, d::GpuUniformInterval, M::CuArray)
    rand!(curand_rng, M)
    map!(M, M) do x
        scale_uniform(x, d.a, d.b)
    end
end

TransformVariables.as(d::GpuUniformInterval) = as(Real, d.a, d.b)

# GpuCircularUniform

struct GpuCircularUniform <: AbstractGpuMeasure end

gpu_measure(::CircularUniform) = GpuCircularUniform()
cpu_measure(::GpuCircularUniform) = CircularUniform()

Base.show(io::IO, ::GpuCircularUniform) = print(io, "GpuCircularUniform")
MeasureTheory.logdensity(::GpuCircularUniform, x) = logdensity(GpuUniformInterval(0, 2π), x)
MeasureTheory.logpdf(d::GpuCircularUniform, x) = logdensity(d, x) - log(2π)

Random.rand!(curand_rng::AbstractRNG, ::GpuCircularUniform, M::CuArray) = rand!(curand_rng, GpuUniformInterval(0, 2π), M)

TransformVariables.as(::GpuCircularUniform) = as○

# GpuBinaryMixture

struct GpuBinaryMixture{T<:AbstractGpuMeasure,U<:AbstractGpuMeasure} <: AbstractGpuMeasure
    c1::T
    c2::U
    log_w1::Float32
    log_w2::Float32
    GpuBinaryMixture(c1::T, c2::U, w1::Real, w2::Real) where {T,U} = new{T,U}(c1, c2, Float32(log(w1 / (w1 + w2))), Float32(log(w2 / (w1 + w2))))
end
GpuBinaryMixture(d::BinaryMixture) = GpuBinaryMixture(gpu_measure(d.c1), gpu_measure(d.c2), exp(d.log_w1), exp(d.log_w2))

gpu_measure(d::BinaryMixture) = GpuBinaryMixture(d)
cpu_measure(d::GpuBinaryMixture) = BinaryMixture(cpu_measure(d.c1), cpu_measure(d.c2), exp(d.log_w1), exp(d.log_w2))

Base.show(io::IO, d::GpuBinaryMixture) = print(io, "GpuBinaryMixture\n  components: $(d.c1), $(d.c2) \n  log weights: $(d.log_w1), $(d.log_w2)")

MeasureTheory.logdensity(d::GpuBinaryMixture, x) = logaddexp(d.log_w1 + logpdf(d.c1, x), d.log_w2 + logpdf(d.c2, x))
MeasureTheory.logpdf(d::GpuBinaryMixture, x) = logdensity(d, x)

function Random.rand!(curand_rng::AbstractRNG, d::GpuBinaryMixture, M::CuArray)
    CURAND.curandGenerateUniform(curand_rng, M, length(M))
    M .= log.(M)
    c1_ind = M .< d.log_w1
    M[c1_ind] .= rand(d.c1, count(c1_ind .> 0))
    c2_ind = .!c1_ind
    M[c2_ind] .= rand(d.c2, count(c2_ind .> 0))
    M
end

# TODO does it make sense to introduce TransformVariables.as for Mixtures?

# GpuProductMeasure
struct GpuProductMeasure{N,T<:AbstractGpuMeasure} <: AbstractGpuMeasure
    internal::T
    size::NTuple{N,Int64}
end
# GpuProductMeasure(d::AbstractGpuMeasure, dims...) = GpuProductMeasure(d, (dims...,))

gpu_measure(d::ProductMeasure) = GpuProductMeasure(marginals(d) |> first |> gpu_measure, size(d))
cpu_measure(d::GpuProductMeasure) = MeasureBase.productmeasure(identity, fill(d.internal |> cpu_measure, d.size))

Base.show(io::IO, d::GpuProductMeasure) = print(io, "GpuProductMeasure\n  internal: $(d.internal) \n  size: $(d.size)")

function reduce_to_last_dim(op, M::AbstractArray{<:Any,N}) where {N}
    R = reduce(op, M; dims=(1:N-1...,))
    dropdims(R; dims=(1:N-1...,))
end

# TODO Interface: all other GpuMeasures require broadcasting, here it is done internally. Is this expected? Will Tensors always be wrapped?
function MeasureTheory.logdensity(d::GpuProductMeasure, x)
    ℓ = logdensity.((d.internal,), x)
    reduce_to_last_dim(+, ℓ)
end
function MeasureTheory.logpdf(d::GpuProductMeasure, x)
    ℓ = logpdf.((d.internal,), x)
    reduce_to_last_dim(+, ℓ)
end

# TODO different sizes of d and M does not make sense, this is a friendly behavior which avoids crashes but hide errors.
Random.rand!(d::GpuProductMeasure, M::CuArray) = CUDA.rand!(CURAND.default_rng(), d.internal, M)
Random.rand(rng::AbstractRNG, d::GpuProductMeasure) = rand(rng, d.internal, d.size)
Random.rand(d::GpuProductMeasure) = rand(d.internal, d.size...)

TransformVariables.as(d::GpuProductMeasure) = as(d.internal)
