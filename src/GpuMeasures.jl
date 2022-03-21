# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using CUDA
using MCMCDepth
using MeasureTheory
using LogExpFunctions
using Random

# TODO Implement Random.rand and return CuArray or keep CUDA.rand separated?

# isbitstype types which correspond to MeasureTheory measures
# Interface: Implement gpu_measure(measure)

struct GpuNormal <: AbstractMeasure
    μ::Float32
    σ::Float32
end

GpuNormal(::Normal{()}) = GpuNormal(0.0, 1.0)
GpuNormal(d::Normal{(:μ, :σ)}) = GpuNormal(d.μ, d.σ)

gpu_measure(d::Normal) = GpuNormal(d)

Base.show(io::IO, d::GpuNormal) = print(io, "GpuNormal, μ: $(d.μ), σ: $(d.σ)")

function MeasureTheory.logdensity(d::GpuNormal, x)
    μ = d.μ
    σ² = d.σ^2
    -0.5 * ((x - μ)^2 / σ²)
end

MeasureTheory.logpdf(d::GpuNormal, x) = logdensity(d, x) - log(d.σ) - log(sqrt(2 * pi))

CUDA.rand(d::GpuNormal, dims::Integer...) = CUDA.randn(dims...; mean=d.μ, stddev=d.σ)
function CUDA.rand!(curand_rng::AbstractRNG, d::GpuNormal, M::CuArray)
    CURAND.curandGenerateNormal(curand_rng, M, length(M), d.μ, d.σ)
    M
end
CUDA.rand!(d::GpuNormal, M::CuArray) = CUDA.rand!(CURAND.default_rng(), d, M)

# GpuExponential
struct GpuExponential <: AbstractMeasure
    λ::Float32
end

GpuExponential(::Exponential{()}) = GpuExponential(1.0)
GpuExponential(d::Exponential{(:λ,)}) = GpuExponential(d.λ)
GpuExponential(d::Exponential{(:β,)}) = GpuExponential(1 / d.β)

gpu_measure(d::Exponential) = GpuExponential(d)

Base.show(io::IO, d::GpuExponential) = print(io, "GpuExponential, λ: $(d.λ)")

MeasureTheory.logdensity(d::GpuExponential, x) = -d.λ * x
MeasureTheory.logpdf(d::GpuExponential, x) = logdensity(d, x) + log(d.λ)

CUDA.rand(d::GpuExponential, dims::Integer...) = log.(CUDA.rand(dims...)) / (-d.λ)
function CUDA.rand!(curand_rng::AbstractRNG, d::GpuExponential, M::CuArray)
    CURAND.curandGenerateUniform(curand_rng, M, length(M))
    map!(x -> log(x) / (-d.λ), M, M)
    M
end
CUDA.rand!(d::GpuExponential, M::CuArray) = CUDA.rand!(CURAND.default_rng(), d, M)

# GpuUniformInterval & CircularUniform

struct GpuUniformInterval <: AbstractMeasure
    a::Float32
    b::Float32
end

GpuUniformInterval(::UniformInterval{()}) = GpuUniformInterval(0.0, 1.0)
GpuUniformInterval(d::UniformInterval{(:a, :b)}) = GpuUniformInterval(d.a, d.b)

gpu_measure(d::UniformInterval) = GpuUniformInterval(d)
gpu_measure(d::CircularUniform) = GpuUniformInterval(0, 2π)

Base.show(io::IO, d::GpuUniformInterval) = print(io, "GpuUniformInterval, a: $(d.a), b: $(d.b)")

MeasureTheory.logdensity(d::GpuUniformInterval, x) = d.a <= x <= d.b ? 0.0 : -Inf
MeasureTheory.logpdf(d::GpuUniformInterval, x) = logdensity(d, x) - log(d.b - d.a)

CUDA.rand(d::GpuUniformInterval, dims::Integer...) = CUDA.rand(dims...) .* (d.b - d.a) .+ d.a
function CUDA.rand!(curand_rng::AbstractRNG, d::GpuUniformInterval, M::CuArray)
    CURAND.curandGenerateUniform(curand_rng, M, length(M))
    map!(x -> x * (d.b - d.a) + d.a, M, M)
    M
end
CUDA.rand!(d::GpuUniformInterval, M::CuArray) = CUDA.rand!(CURAND.default_rng(), d, M)

# GpuBinaryMixture

struct GpuBinaryMixture{T<:AbstractMeasure,U<:AbstractMeasure} <: AbstractMeasure
    c1::T
    c2::U
    log_w1::Float32
    log_w2::Float32
end

GpuBinaryMixture(d::BinaryMixture) = GpuBinaryMixture(gpu_measure(d.c1), gpu_measure(d.c2), Float32(d.log_w1), Float32(d.log_w2))

gpu_measure(d::BinaryMixture) = GpuBinaryMixture(d)

Base.show(io::IO, d::GpuBinaryMixture) = print(io, "GpuBinaryMixture\n  components: $(d.c1), $(d.c2) \n  log weights: $(d.log_w1), $(d.log_w2)")

MeasureTheory.logdensity(d::GpuBinaryMixture, x) = logaddexp(d.log_w1 + logpdf(d.c1, x), d.log_w2 + logpdf(d.c2, x))
MeasureTheory.logpdf(d::GpuBinaryMixture, x) = logdensity(d, x)

function CUDA.rand(d::GpuBinaryMixture, dims::Integer...)
    # TODO performance sucks, but probably not used in performance critical code
    M = CUDA.rand(dims...) .|> log
    C1 = CUDA.rand(d.c1, dims...)
    C2 = CUDA.rand(d.c2, dims...)
    c1_ind = M .< d.log_w1
    M[c1_ind] .= C1[c1_ind]
    c2_ind = .!c1_ind
    M[c2_ind] .= C2[c2_ind]
    M
end

# Test GpuNormal
gn = Normal(1.0, 2.0) |> gpu_measure
M = CUDA.rand(gn, 100, 100)
CUDA.rand!(gn, M)
logpdf.((gn,), M)
logpdf(gn, 1.0) ≈ logpdf(Normal(1.0, 2.0), 1.0)

# Test GpuExponential
ge = Exponential(0.1) |> gpu_measure
M = CUDA.rand(ge, 100, 100)
CUDA.rand!(ge, M)
GE = CUDA.fill(ge, size(M))
logpdf.(GE, M)
logpdf(ge, 1.0) ≈ logpdf(Exponential(0.1), 1.0)

# Test GpuBinaryMixture
gbm = BinaryMixture(Normal(1.0, 2.0), Normal(10.0, 0.1), 0.1, 0.9) |> gpu_measure
M = CUDA.rand(gbm, 100, 100)
logpdf.((gbm,), M)
logpdf(gbm, 1.0) ≈ logpdf(BinaryMixture(Normal(1.0, 2.0), Normal(10.0, 0.1), 0.1, 0.9), 1.0)

# Test GpuUniformInterval
gu = UniformInterval(1.0, 10.0) |> gpu_measure
M = CUDA.rand(gu, 100, 100)
CUDA.rand!(gu, M)
MeasureTheory.logpdf.((gu,), M)
logpdf(gu, 0.5) == logpdf(UniformInterval(1.0, 10.0), 0.5)
isapprox(logpdf(gu, 1.5), logpdf(UniformInterval(1.0, 10.0), 1.5); atol=0.0000001)
