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

"""
    rand!(d, M)
Mutate `M` with random samples from the measure `d` using the CUDA default RNG.
"""
Random.rand!(d::AbstractGpuMeasure, M::CuArray) = rand!(CUDA.RNG(), d, M)
"""
    rand(rng, T, d, dim, dims...)
Return a CUDA array of type `T` with `dim` dimensions and `dims` dimensions sampled from the measure `d`.
"""
Base.rand(rng::AbstractRNG, ::Type{T}, d::AbstractGpuMeasure, dim::Integer=1, dims::Integer...) where {T} = rand!(rng, d, CuArray{T}(undef, dim, dims...))
"""
    rand(T, d, dim, dims...)
Return a CUDA array of type `T` with `dim` dimensions and `dims` dimensions sampled from the measure `d` using the CUDA default RNG.
"""
Base.rand(::Type{T}, d::AbstractGpuMeasure, dim::Integer=1, dims::Integer...) where {T} = rand(CUDA.RNG(), T, d, dim, dims...)
"""
    rand(T, d, dim, dims...)
Return a CUDA array of type `Float32` with `dim` dimensions and `dims` dimensions sampled from the measure `d` using the CUDA default RNG.
"""
Base.rand(d::AbstractGpuMeasure, dim::Integer=1, dims::Integer...) = rand(Float32, d, dim, dims...)

# GpuNormal

struct GpuNormal <: AbstractGpuMeasure
    μ::Float32
    σ::Float32
end

GpuNormal() = GpuNormal(0.0, 1.0)
GpuNormal(::Normal{()}) = GpuNormal()
GpuNormal(d::Normal{(:μ, :σ)}) = GpuNormal(d.μ, d.σ)

gpu_measure(d::Normal) = GpuNormal(d)
cpu_measure(d::GpuNormal) = Normal(d.μ, d.σ)

Base.show(io::IO, d::GpuNormal) = print(io, "GpuNormal, μ: $(d.μ), σ: $(d.σ)")

function MeasureTheory.logdensity(d::GpuNormal, x::T) where {T<:Real}
    μ = d.μ
    σ² = d.σ^2
    -T(0.5) * ((x - μ)^2 / σ²)
end

MeasureTheory.logpdf(d::GpuNormal, x::T) where {T<:Real} = logdensity(d, x) - log(d.σ) - log(sqrt(T(2π)))

scale_shift_normal(x::Real, μ::Real, σ::Real) = σ * x + μ

function Random.rand!(rng::AbstractRNG, d::GpuNormal, M::CuArray)
    randn!(rng, M)
        map!(M, M) do x
        scale_shift_normal(x, d.μ, d.σ)
    end
end

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

MeasureTheory.logdensity(d::GpuExponential, x::Real) = -d.λ * x
MeasureTheory.logpdf(d::GpuExponential, x::Real) = logdensity(d, x) + log(d.λ)

uniform_to_exp(x::Real, λ::Real) = log(x) / (-λ)

function Random.rand!(rng::AbstractRNG, d::GpuExponential, M::CuArray)
    rand!(rng, M)
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

MeasureTheory.logdensity(d::GpuUniformInterval, x::T) where {T<:Real} = d.a <= x <= d.b ? zero(T) : -typemax(T)
MeasureTheory.logpdf(d::GpuUniformInterval, x::Real) = logdensity(d, x) - log(d.b - d.a)

scale_uniform(x, a, b) = x * (b - a) + a

function Random.rand!(rng::AbstractRNG, d::GpuUniformInterval, M::CuArray)
    rand!(rng, M)
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
MeasureTheory.logdensity(::GpuCircularUniform, x::Real) = logdensity(GpuUniformInterval(0, 2π), x)
MeasureTheory.logpdf(d::GpuCircularUniform, x::Real) = logdensity(d, x) - log(2π)

Random.rand!(rng::AbstractRNG, ::GpuCircularUniform, M::CuArray) = rand!(rng, GpuUniformInterval(0, 2π), M)

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

MeasureTheory.logdensity(d::GpuBinaryMixture, x::Real) = logaddexp(d.log_w1 + logpdf(d.c1, x), d.log_w2 + logpdf(d.c2, x))
MeasureTheory.logpdf(d::GpuBinaryMixture, x::Real) = logdensity(d, x)

function Random.rand!(rng::AbstractRNG, d::GpuBinaryMixture, M::CuArray{T}) where {T}
    rand!(rng, M)
    M .= log.(M)
    c1_ind = M .< d.log_w1
    M[c1_ind] .= rand(rng, T, d.c1, count(c1_ind .> 0))
    c2_ind = .!c1_ind
    M[c2_ind] .= rand(rng, T, d.c2, count(c2_ind .> 0))
    M
end

# TODO does it make sense to introduce TransformVariables.as for Mixtures?

# GpuProductMeasure

struct GpuProductMeasure{N,T<:AbstractGpuMeasure} <: AbstractGpuMeasure
    internal::T
    size::NTuple{N,Int64}
end
GpuProductMeasure(d::AbstractGpuMeasure, dims...) = GpuProductMeasure(d, dims)
GpuProductMeasure(d::ProductMeasure) = GpuProductMeasure(marginals(d) |> first |> gpu_measure, size(d))

gpu_measure(d::ProductMeasure) = GpuProductMeasure(d)
cpu_measure(d::GpuProductMeasure) = MeasureBase.productmeasure(identity, fill(d.internal |> cpu_measure, d.size))

Base.show(io::IO, d::GpuProductMeasure) = print(io, "GpuProductMeasure\n  internal: $(d.internal) \n  size: $(d.size)")

Base.size(d::GpuProductMeasure) = d.size

function MeasureTheory.logdensity(d::GpuProductMeasure, x::CuArray)
    ℓ = logdensity.((d.internal,), x)
    sum(ℓ)
end
function MeasureTheory.logpdf(d::GpuProductMeasure, x::CuArray)
    ℓ = logpdf.((d.internal,), x)
    sum(ℓ)
end

# TODO different sizes of d and M does not make sense, this is a friendly behavior which avoids crashes but hide errors.
function Random.rand!(d::GpuProductMeasure, M::CuArray)
    if d.size != size(M)
        @warn "Dimension mismatch of GpuProductMeasure and Array: $(d.size) != $(size(M))"
    end
    rand!(CUDA.RNG(), d.internal, M)
end
Random.rand(rng::AbstractRNG, T::Type, d::GpuProductMeasure) = rand(rng, T, d.internal, d.size...)
Random.rand(T::Type, d::GpuProductMeasure) = rand(T, d.internal, d.size...)
Random.rand(d::GpuProductMeasure) = rand(d.internal, d.size...)

TransformVariables.as(d::GpuProductMeasure) = as(d.internal)

# GpuVectorizedMeasure

"""
    GpuVectorizedMeasure
Behaves similar to the ProductMeasure but assumes a vectorization of the data over last dimension.
"""
struct GpuVectorizedMeasure{N,T<:AbstractGpuMeasure} <: AbstractGpuMeasure
    internal::T
    size::NTuple{N,Int64}
end
GpuVectorizedMeasure(d::AbstractGpuMeasure, dims...) = GpuVectorizedMeasure(d, dims)
GpuVectorizedMeasure(d::VectorizedMeasure) = GpuVectorizedMeasure(d.internal |> gpu_measure, size(d))

gpu_measure(d::VectorizedMeasure) = GpuVectorizedMeasure(d)
cpu_measure(d::GpuVectorizedMeasure) = VectorizedMeasure(d.internal |> cpu_measure, d.size)

Base.show(io::IO, d::GpuVectorizedMeasure) = print(io, "GpuVectorizedMeasure\n  internal: $(d.internal) \n  size: $(d.size)")

Base.size(d::GpuVectorizedMeasure) = d.size

function MeasureTheory.logdensity(d::GpuVectorizedMeasure, x::CuArray)
    ℓ = logdensity.((d.internal,), x)
    reduce_to_last_dim(+, ℓ)
end
function MeasureTheory.logpdf(d::GpuVectorizedMeasure, x::CuArray)
    ℓ = logpdf.((d.internal,), x)
    reduce_to_last_dim(+, ℓ)
end

function Random.rand!(d::GpuVectorizedMeasure, M::CuArray)
    if d.size != size(M)
        @warn "Dimension mismatch of GpuVectorizedMeasure and Array: $(d.size) != $(size(M))"
    end
    rand!(CUDA.RNG(), d.internal, M)
end
Random.rand(rng::AbstractRNG, T::Type, d::GpuVectorizedMeasure) = rand(rng, T, d.internal, d.size...)
Random.rand(T::Type, d::GpuVectorizedMeasure) = rand(T, d.internal, d.size...)
Random.rand(d::GpuVectorizedMeasure) = rand(d.internal, d.size...)

TransformVariables.as(d::GpuVectorizedMeasure) = as(d.internal)
