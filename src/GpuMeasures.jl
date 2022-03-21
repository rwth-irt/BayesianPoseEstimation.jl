# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using CUDA
using MCMCDepth
using MeasureTheory
using LogExpFunctions
using Random

# isbitstype types which correspond to MeasureTheory measures
# Interface: Implement gpu_measure(cpu_measure) & cpu_measure(gpu_measure)

struct GpuNormal <: AbstractMeasure
    μ::Float32
    σ::Float32
end

GpuNormal(::Normal{()}) = GpuNormal(0.0, 1.0)
GpuNormal(d::Normal{(:μ, :σ)}) = GpuNormal(d.μ, d.σ)

gpu_measure(d::Normal) = GpuNormal(d)
cpu_measure(d::GpuNormal) = Normal(d.μ, d.σ)

Base.show(io::IO, d::GpuNormal) = print(io, "GpuNormal, μ: $(d.μ), σ: $(d.σ)")

function MeasureTheory.logdensity(d::GpuNormal, x)
    μ = d.μ
    σ² = d.σ^2
    -0.5 * ((x - μ)^2 / σ²)
end

MeasureTheory.logpdf(d::GpuNormal, x) = logdensity(d, x) - log(d.σ) - log(sqrt(2 * pi))

Base.rand(d::GpuNormal, dims::Integer...) = CUDA.randn(dims...; mean=d.μ, stddev=d.σ)
function Random.rand!(curand_rng::AbstractRNG, d::GpuNormal, M::CuArray)
    CURAND.curandGenerateNormal(curand_rng, M, length(M), d.μ, d.σ)
    M
end
Random.rand!(d::GpuNormal, M::CuArray) = CUDA.rand!(CURAND.default_rng(), d, M)

# GpuExponential
struct GpuExponential <: AbstractMeasure
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

Base.rand(d::GpuExponential, dims::Integer...) = log.(CUDA.rand(dims...)) / (-d.λ)
function Random.rand!(curand_rng::AbstractRNG, d::GpuExponential, M::CuArray)
    CURAND.curandGenerateUniform(curand_rng, M, length(M))
    map!(x -> log(x) / (-d.λ), M, M)
    M
end
Random.rand!(d::GpuExponential, M::CuArray) = CUDA.rand!(CURAND.default_rng(), d, M)

# GpuUniformInterval & CircularUniform

struct GpuUniformInterval <: AbstractMeasure
    a::Float32
    b::Float32
end

GpuUniformInterval(::UniformInterval{()}) = GpuUniformInterval(0.0, 1.0)
GpuUniformInterval(d::UniformInterval{(:a, :b)}) = GpuUniformInterval(d.a, d.b)

gpu_measure(d::UniformInterval) = GpuUniformInterval(d)
cpu_measure(d::GpuUniformInterval) = UniformInterval(d.a, d.b)
# TODO required?
gpu_measure(::CircularUniform) = GpuUniformInterval(0, 2π)

Base.show(io::IO, d::GpuUniformInterval) = print(io, "GpuUniformInterval, a: $(d.a), b: $(d.b)")

MeasureTheory.logdensity(d::GpuUniformInterval, x) = d.a <= x <= d.b ? 0.0 : -Inf
MeasureTheory.logpdf(d::GpuUniformInterval, x) = logdensity(d, x) - log(d.b - d.a)

Base.rand(d::GpuUniformInterval, dims::Integer...) = CUDA.rand(dims...) .* (d.b - d.a) .+ d.a
function Random.rand!(curand_rng::AbstractRNG, d::GpuUniformInterval, M::CuArray)
    CURAND.curandGenerateUniform(curand_rng, M, length(M))
    map!(x -> x * (d.b - d.a) + d.a, M, M)
    M
end
Random.rand!(d::GpuUniformInterval, M::CuArray) = CUDA.rand!(CURAND.default_rng(), d, M)

# GpuBinaryMixture

struct GpuBinaryMixture{T<:AbstractMeasure,U<:AbstractMeasure} <: AbstractMeasure
    c1::T
    c2::U
    log_w1::Float32
    log_w2::Float32
end

GpuBinaryMixture(d::BinaryMixture) = GpuBinaryMixture(gpu_measure(d.c1), gpu_measure(d.c2), Float32(d.log_w1), Float32(d.log_w2))

gpu_measure(d::BinaryMixture) = GpuBinaryMixture(d)
cpu_measure(d::GpuBinaryMixture) = BinaryMixture(cpu_measure(d.c1), cpu_measure(d.c2), exp(d.log_w1), exp(d.log_w2))

Base.show(io::IO, d::GpuBinaryMixture) = print(io, "GpuBinaryMixture\n  components: $(d.c1), $(d.c2) \n  log weights: $(d.log_w1), $(d.log_w2)")

MeasureTheory.logdensity(d::GpuBinaryMixture, x) = logaddexp(d.log_w1 + logpdf(d.c1, x), d.log_w2 + logpdf(d.c2, x))
MeasureTheory.logpdf(d::GpuBinaryMixture, x) = logdensity(d, x)

function Base.rand(d::GpuBinaryMixture, dims::Integer...)
    M = CuArray{Float32}(undef, dims...)
    rand!(d, M)
    M
end

function Random.rand!(curand_rng::AbstractRNG, d::GpuBinaryMixture, M::CuArray)
    CURAND.curandGenerateUniform(curand_rng, M, length(M))
    M .= log.(M)
    c1_ind = M .< d.log_w1
    M[c1_ind] .= rand(d.c1, count(c1_ind .> 0))
    c2_ind = .!c1_ind
    M[c2_ind] .= rand(d.c2, count(c2_ind .> 0))
    M
end
Random.rand!(d::GpuBinaryMixture, M::CuArray) = CUDA.rand!(CURAND.default_rng(), d, M)
