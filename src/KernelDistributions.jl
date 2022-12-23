# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Bijectors
using CUDA
using DensityInterface
using Distributions
using MCMCDepth
using LogExpFunctions
using Logging
using Random
using StatsFuns: normlogcdf, norminvlogcdf, xval, zval


# TODO At one point most of the distributions could be replaced with Distributions.jl. Mixtures could be problematic.
# TODO should open a pull request to fix type of https://github.com/JuliaStats/Distributions.jl/blob/d19ac4526bab2584a84323eea4af92805f99f034/src/univariate/continuous/uniform.jl#L120

"""
MeasureTheory.jl is what I have used because of the nicer interface until now, but all the type are not isbits and can not be used on the GPU.
Distributions.jl is pretty close but not perfect for the execution on the GPU:
- Mostly type stable
- Mixtures a quirky
- Uniform is not strongly typed resulting in Float64 calculations all the time.

Here, I provide stripped-down Distributions which are isbitstype, strongly typed and thus support execution on the GPU.
KernelDistributions offer the following interface functions:
- `DensityInterface.logdensityof(dist::KernelDistribution, x)`
- `Random.rand!(rng, dist::KernelDistribution, A)`
- `Base.rand(rng, dist::KernelDistribution, dims...)`
- `Base.eltype(::Type{<:AbstractKernelDistribution})`: Number format of the distribution, e.g. Float16

The Interface requires the following to be implemented:
- Bijectors.bijector(d): Bijector
- `Base.rand(rng, dist::MyKernelDistribution{T})::T` generate a single random number from the distribution
- `Distributions.logpdf(dist::MyKernelDistribution{T}, x)::T` evaluate the normalized logdensity
- `Base.maximum(d), Base.minimum(d), Distributions.insupport(d)`: Determine the support of the distribution
- `Distributions.logcdf(d, x), Distributions.invlogcdf(d, x)`: Support for Truncated{D}

Most of the time Float64 precision is not required, especially for GPU computations.
Thus, we default to Float32, mostly for memory capacity reasons.
"""
abstract type AbstractKernelDistribution{T,S<:ValueSupport} <: UnivariateDistribution{S} end

# WARN parametric alias causes method ambiguities, since the parametric type is always present
const KernelOrTransformedKernel = Union{AbstractKernelDistribution,UnivariateTransformed{<:AbstractKernelDistribution},Truncated{<:AbstractKernelDistribution}}

const KernelOrKernelArray = Union{KernelOrTransformedKernel,AbstractArray{<:KernelOrTransformedKernel}}

"""
    eltype(kernel_distribution)
Get the parameter type of the distribution, e.g. Float16
"""
Base.eltype(::Type{<:AbstractKernelDistribution{T}}) where {T} = T
Base.eltype(::Type{<:UnivariateTransformed{T}}) where {T} = eltype(T)

# Make KernelDistributions extendable by allowing to override it for custom distributions
array_for_rng(rng::AbstractRNG, dist::KernelOrTransformedKernel, dims::Integer...) = array_for_rng(rng, eltype(dist), dims...)
array_for_rng(rng::AbstractRNG, ::AbstractArray{T}, dims::Integer...) where {T<:KernelOrTransformedKernel} = array_for_rng(rng, eltype(T), dims...)


# By default, Distributions.jl disallows logdensityof with multiple samples (Arrays and Matrices). KernelDistributions should be inherently allowing multiple samples.
DensityInterface.logdensityof(dist::KernelOrKernelArray, x::AbstractArray) = logpdf.(dist, x)
DensityInterface.logdensityof(dist::KernelOrKernelArray, x::AbstractMatrix) = logpdf.(dist, x)

# Random Interface

"""
    rand(rng, dist, [dims...])
Sample an Array from `dist` of size `dims`.
"""
function Base.rand(rng::AbstractRNG, dist::KernelOrTransformedKernel, dims::Int64...)
    A = array_for_rng(rng, dist, dims...)
    rand!(rng, dist, A)
end

# Avoid recursions of the above
Base.rand(rng::AbstractRNG, transformed_dist::UnivariateTransformed{<:AbstractKernelDistribution}) = transformed_dist.transform(rand(rng, transformed_dist.dist))

# Distributions.jl implementation won't run on the GPU. Only use the most general case, which might be slower but more robust
Base.rand(rng::AbstractRNG, dist::Truncated{<:AbstractKernelDistribution{T}}) where {T} = invlogcdf(dist.untruncated, logaddexp(T(dist.loglcdf), T(dist.logtp) - randexp(rng, T)))

# Arrays of KernelDistributions → sample from the distributions instead of selecting random elements of the array

"""
    rand(rng, dist, [dims...])
Sample an Array from `dists` of size `dims`.
"""
function Base.rand(rng::AbstractRNG, dists::AbstractArray{<:KernelOrTransformedKernel}, dims::Integer...)
    A = array_for_rng(rng, dists, size(dists)..., dims...)
    rand!(rng, dists, A)
end

"""
    rand!(rng, dist, A)
Mutate the array A by sampling from `dist`.
"""
Random.rand!(rng::AbstractRNG, dist::AbstractArray{<:KernelOrTransformedKernel}, A::AbstractArray) = _rand!(rng, dist, A)
Random.rand!(rng::AbstractRNG, dist::KernelOrTransformedKernel, A::AbstractArray) = _rand!(rng, dist, A)
# Resolve ambiguities with Distributions.jl
Random.rand!(rng::AbstractRNG, dist::KernelOrTransformedKernel, A::AbstractArray{<:Real}) = _rand!(rng, dist, A)

# CPU implementation

"""
    _rand!(rng, dist, A)
Internal inplace random function which allows dispatching based on the RNG and output array.
Keeping dist as Any  allows more flexibility, for example passing a Broadcasted to avoid allocations.
"""
function _rand!(rng::AbstractRNG, dist, A::AbstractArray)
    # Avoid endless recursions for rand(rng, dist::KernelOrTransformedKernel)
    @. A = rand(rng, dist)
end

# GPU implementation

# Currently only the CUDA.RNG is supported in Kernels.
function _rand!(rng::CUDA.RNG, dist, A::CuArray)
    _rand_cuda_rng!(dist, A, rng.seed, rng.counter)
    new_counter = Int64(rng.counter) + length(A)
    overflow, remainder = fldmod(new_counter, typemax(UInt32))
    rng.seed += overflow
    rng.counter = remainder
    return A
end

# Function barrier for CUDA.RNG which is not isbits.
# Wrapping rng in Tuple for broadcasting does not work → anonymous function is the workaround 
# Thanks vchuravy https://github.com/JuliaGPU/CUDA.jl/issues/1480#issuecomment-1102245813
function _rand_cuda_rng!(dist, A, seed, counter)
    A .= (x -> rand(device_rng(seed, counter), x)).(dist)
end

"""
    device_rng(seed, counter)
Use it inside a kernel to generate a correctly seeded device RNG.
"""
function device_rng(seed, counter)
    # Replaced during kernel compilation: https://github.com/JuliaGPU/CUDA.jl/blob/778f7fa21f3f73841a2dada57767e358f80e5997/src/device/random.jl#L37
    rng = Random.default_rng()
    # Same as in CUDA.jl: https://github.com/JuliaGPU/CUDA.jl/blob/778f7fa21f3f73841a2dada57767e358f80e5997/src/random.jl#L79
    @inbounds Random.seed!(rng, seed, counter)
    rng
end

# Bijector for arrays

Bijectors.bijector(dists::AbstractArray{<:KernelOrTransformedKernel}) = bijector(first(dists))

# KernelNormal

struct KernelNormal{T<:Real} <: AbstractKernelDistribution{T,Continuous}
    μ::T
    σ::T
end
KernelNormal(μ::Real, σ::Real) = KernelNormal(promote(μ, σ)...)
KernelNormal(::Type{T}=Float32) where {T} = KernelNormal{T}(0.0, 1.0)

Base.show(io::IO, dist::KernelNormal{T}) where {T} = print(io, "KernelNormal{$(T)}, μ: $(dist.μ), σ: $(dist.σ)")

function Distributions.logpdf(dist::KernelNormal{T}, x) where {T}
    μ = dist.μ
    σ² = dist.σ^2
    # Unnormalized like MeasureTheroy logdensity_def
    ℓ = -T(0.5) * ((T(x) - μ)^2 / σ²)
    ℓ - log(dist.σ) - log(sqrt(T(2π)))
end

Base.rand(rng::AbstractRNG, dist::KernelNormal{T}) where {T} = dist.σ * randn(rng, T) + dist.μ

Base.maximum(::KernelNormal{T}) where {T} = typemax(T)
Base.minimum(::KernelNormal{T}) where {T} = typemin(T)
Bijectors.bijector(::KernelNormal) = ZeroIdentity()
Distributions.insupport(::KernelNormal, ::Real) = true

# Support Truncated{KernelNormal}
Distributions.logcdf(dist::KernelNormal{T}, x::Real) where {T} = normlogcdf(dist.μ, dist.σ, x)
Distributions.invlogcdf(dist::KernelNormal{T}, lp::Real) where {T} = norminvlogcdf(dist.μ, dist.σ, lp)

# KernelExponential

struct KernelExponential{T<:Real} <: AbstractKernelDistribution{T,Continuous}
    θ::T
end
KernelExponential(::Type{T}=Float32) where {T} = KernelExponential{T}(1.0)

Base.show(io::IO, dist::KernelExponential{T}) where {T} = print(io, "KernelExponential{$(T)}, θ: $(dist.θ)")

function Distributions.logpdf(dist::KernelExponential{T}, x) where {T}
    if insupport(dist, x)
        λ = inv(dist.θ)
        -λ * T(x) + log(λ)
    else
        -typemax(T)
    end
end

Base.rand(rng::AbstractRNG, dist::KernelExponential{T}) where {T} = dist.θ * randexp(rng, T)

Base.maximum(::KernelExponential{T}) where {T} = typemax(T)
Base.minimum(::KernelExponential{T}) where {T} = zero(T)
Bijectors.bijector(::KernelExponential) = Bijectors.Log{0}()
Distributions.insupport(dist::KernelExponential, x::Real) = minimum(dist) <= x
# Support Truncated{KernelExponential}
Distributions.logcdf(dist::KernelExponential{T}, x::Real) where {T} = log1mexp(-max(T(x) / dist.θ, zero(T)))
Distributions.invlogcdf(dist::KernelExponential{T}, lp::Real) where {T} = -log1mexp(T(lp)) * dist.θ

# KernelUniform

struct KernelUniform{T<:Real} <: AbstractKernelDistribution{T,Continuous}
    min::T
    max::T
end
KernelUniform(min::Real, max::Real) = KernelUniform(promote(min, max)...)
KernelUniform(::Type{T}=Float32) where {T} = KernelUniform{T}(0.0, 1.0)

Base.show(io::IO, dist::KernelUniform{T}) where {T} = print(io, "KernelUniform{$(T)}, a: $(dist.min), b: $(dist.max)")

Distributions.logpdf(dist::KernelUniform{T}, x) where {T<:Real} = insupport(dist, x) ? -log(dist.max - dist.min) : -typemax(T)

Base.rand(rng::AbstractRNG, dist::KernelUniform{T}) where {T} = (dist.max - dist.min) * rand(rng, T) + dist.min

Base.maximum(dist::KernelUniform) = dist.max
Base.minimum(dist::KernelUniform) = dist.min
Bijectors.bijector(dist::KernelUniform) = Bijectors.TruncatedBijector{0}(minimum(dist), maximum(dist))
Distributions.insupport(dist::KernelUniform, x::Real) = minimum(dist) <= x <= maximum(dist)

# KernelTailUniform

"""
    KernelTailUniform(min, max)
Acts like a uniform distribution of [min,max] but ignores outliers and always returns 1/(max-min) as probability.
"""
struct KernelTailUniform{T<:Real} <: AbstractKernelDistribution{T,Continuous}
    min::T
    max::T
end
KernelTailUniform(min::Real, max::Real) = KernelTailUniform(promote(min, max)...)
KernelTailUniform(::Type{T}=Float32) where {T} = KernelTailUniform{T}(0.0, 1.0)

Base.show(io::IO, dist::KernelTailUniform{T}) where {T} = print(io, "KernelTailUniform{$(T)}, a: $(dist.min), b: $(dist.max)")

Distributions.logpdf(dist::KernelTailUniform{T}, x) where {T<:Real} = -log(dist.max - dist.min)

Base.rand(rng::AbstractRNG, dist::KernelTailUniform{T}) where {T} = (dist.max - dist.min) * rand(rng, T) + dist.min

Base.maximum(::KernelTailUniform{T}) where {T} = typemax(T)
Base.minimum(::KernelTailUniform{T}) where {T} = typemin(T)
Bijectors.bijector(::KernelTailUniform) = ZeroIdentity()
Distributions.insupport(dist::KernelTailUniform, x::Real) = true

# KernelCircularUniform

struct KernelCircularUniform{T<:Real} <: AbstractKernelDistribution{T,Continuous} end

KernelCircularUniform(::Type{T}=Float32) where {T} = KernelCircularUniform{T}()

Base.show(io::IO, ::KernelCircularUniform{T}) where {T} = print(io, "KernelCircularUniform{$(T)}")

Distributions.logpdf(dist::KernelCircularUniform{T}, x) where {T} = insupport(dist, x) ? -log(T(2π)) : -typemax(T)

Base.rand(rng::AbstractRNG, ::KernelCircularUniform{T}) where {T} = T(2π) * rand(rng, T)

Base.maximum(::KernelCircularUniform{T}) where {T} = T(2π)
Base.minimum(::KernelCircularUniform{T}) where {T} = zero(T)
Bijectors.bijector(::KernelCircularUniform) = Circular{0}()
Distributions.insupport(dist::KernelCircularUniform, x::Real) = minimum(dist) <= x <= maximum(dist)

# KernelDirac

struct KernelDirac{T<:Real} <: AbstractKernelDistribution{T,Discrete}
    value::T
end

Base.show(io::IO, dist::KernelDirac{T}) where {T} = print(io, "KernelDirac{$(T)}, value: $(dist.value)")

Distributions.logpdf(dist::KernelDirac{T}, x) where {T<:Real} = insupport(dist, x) ? zero(T) : typemin(T)

Base.rand(::AbstractRNG, dist::KernelDirac) = dist.value

Base.maximum(dist::KernelDirac) = dist.value
Base.minimum(dist::KernelDirac) = dist.value
Bijectors.bijector(::KernelDirac) = ZeroIdentity()
Distributions.insupport(dist::KernelDirac, x::Real) = x == dist.value

# KernelBinaryMixture

# Value support makes only sense to be either Discrete or Continuous
struct KernelBinaryMixture{T<:AbstractFloat,S<:ValueSupport,U<:UnivariateDistribution{S},V<:UnivariateDistribution{S}} <: AbstractKernelDistribution{T,S}
    dist_1::U
    dist_2::V
    # Prefer log here, since the logdensity will be used more often than rand
    log_weight_1::T
    log_weight_2::T
    KernelBinaryMixture{T}(dist_1::U, dist_2::V, weight_1, weight_2) where {T,S<:ValueSupport,U<:UnivariateDistribution{S},V<:UnivariateDistribution{S}} = new{T,S,U,V}(dist_1, dist_2, log(weight_1 / (weight_1 + weight_2)), log(weight_2 / (weight_1 + weight_2)))
end

KernelBinaryMixture(::Type{T}, dist_1, dist_2, weight_1::Real, weight_2::Real) where {T} = KernelBinaryMixture{T}(dist_1, dist_2, weight_1, weight_2)
KernelBinaryMixture(dist_1, dist_2, weight_1::T, weight_2::T) where {T} = KernelBinaryMixture{T}(dist_1, dist_2, weight_1, weight_2)


Base.show(io::IO, dist::KernelBinaryMixture{T}) where {T} = print(io, "KernelBinaryMixture{$(T)}\n  components: $(dist.dist_1), $(dist.dist_2) \n  log weights: $(dist.log_weight_1), $(dist.log_weight_2)")

Distributions.logpdf(dist::KernelBinaryMixture{T}, x) where {T} = insupport(dist, x) ? logaddexp(dist.log_weight_1 + logdensityof(dist.dist_1, x), dist.log_weight_2 + logdensityof(dist.dist_2, x)) : -typemax(T)


function Base.rand(rng::AbstractRNG, dist::KernelBinaryMixture{T}) where {T}
    log_u = log(rand(rng, T))
    if log_u < dist.log_weight_1
        rand(rng, dist.dist_1)
    else
        rand(rng, dist.dist_2)
    end
end

# The support of a mixture is the union of the support of its components
Base.maximum(dist::KernelBinaryMixture) = max(maximum(dist.dist_1), maximum(dist.dist_2))
Base.minimum(dist::KernelBinaryMixture) = min(minimum(dist.dist_1), minimum(dist.dist_2))
Bijectors.bijector(dist::KernelBinaryMixture) = Bijectors.TruncatedBijector(minimum(dist), maximum(dist))
Distributions.insupport(dist::KernelBinaryMixture, x::Real) = minimum(dist) <= x <= maximum(dist)
