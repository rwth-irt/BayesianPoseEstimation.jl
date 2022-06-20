# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Bijectors
using CUDA
using DensityInterface
using MCMCDepth
using LogExpFunctions
using Logging
using Random


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
- maximum(d), minimum(d), insupport(d): Determine the support of the distribution

The Interface requires the following to be implemented:
- Bijectors.bijector(d): Bijector
- `Base.rand(rng, dist::MyKernelDistribution{T})::T` generate a single random number from the distribution
TODO
- `DensityInterface.logdensityof(dist::MyKernelDistribution{T}, x)::T` evaluate the normalized logdensity

Most of the time Float64 precision is not required, especially for GPU computations.
Thus, we default to Float32, mostly for memory capacity reasons.
"""
abstract type AbstractKernelDistribution{T} end

# WARN parametric alias causes method ambiguities, since the parametric type is always present
const KernelOrKernelArray = Union{AbstractKernelDistribution,AbstractArray{<:AbstractKernelDistribution}}
# DensityInterface

@inline DensityInterface.DensityKind(::AbstractKernelDistribution) = IsDensity()

# A single distribution should behave similar to a 0 dimensional array
Base.broadcastable(x::AbstractKernelDistribution) = Ref(x)

# Random interface

"""
    rand!(rng, dist, A)
Mutate the array A by sampling from `dist`.
"""
Random.rand!(rng::AbstractRNG, dist::KernelOrKernelArray, A::AbstractArray) = _rand!(rng, dist, A)

"""
    rand(rng, dist, [dims...])
Sample an Array from `dist` of size `dims`.
"""
function Base.rand(rng::AbstractRNG, dist::AbstractKernelDistribution{T}, dims::Integer...) where {T}
    A = array_for_rng(rng, T, dims...)
    rand!(rng, dist, A)
end

"""
    rand(rng, dist, [dims...])
Sample an Array from `dists` of size `dims`.
"""
function Base.rand(rng::AbstractRNG, dists::AbstractArray{<:AbstractKernelDistribution{T}}, dims::Integer...) where {T}
    A = array_for_rng(rng, T, size(dists)..., dims...)
    rand!(rng, dists, A)
end

# TEST removed GLOBAL_RNG methods

# Bijector for arrays
Bijectors.bijector(dists::AbstractArray{<:AbstractKernelDistribution}) = bijector(first(dists))

# CPU implementation

# TEST
function _rand!(rng::AbstractRNG, dist::KernelOrKernelArray, A::Array)
    A .= rand.(rng, dist)
end

# GPU implementation

# TODO Might want this to fail instead of fallback?
function _rand!(rng::AbstractRNG, dist::KernelOrKernelArray, A::CuArray{T}) where {T}
    @warn "Using unsupported RNG of type $(typeof(rng)) on CuArray. Falling back to SLOW sampling on CPU and copying it to the GPU."
    copyto!(A, rand!(rng, dist, Array{T}(undef, size(A)...)))
end

function _rand!(rng::CUDA.RNG, dist::KernelOrKernelArray, A::CuArray)
    d = maybe_cuda(A, dist)
    rand_barrier(d, A, rng.seed, rng.counter)
    new_counter = Int64(rng.counter) + length(A)
    overflow, remainder = fldmod(new_counter, typemax(UInt32))
    rng.seed += overflow
    rng.counter = remainder
    return A
end

# Function barrier for CUDA.RNG which is not isbits.
# Wrapping rng in Tuple for broadcasting does not work → anonymous function is the workaround 
# Thanks vchuravy https://github.com/JuliaGPU/CUDA.jl/issues/1480#issuecomment-1102245813
function rand_barrier(dist::AbstractKernelDistribution, A::CuArray, seed, counter)
    A .= (x -> rand(device_rng(seed, counter), x)).((dist,))
end

function rand_barrier(dists::CuArray{<:AbstractKernelDistribution}, A::CuArray, seed, counter)
    A .= (x -> rand(device_rng(seed, counter), x)).(dists)
end

# GPU transfer helpers

"""
    array_for_rng(rng, T, dims...)
Generate the correct array to be used in rand! based on the rng provided.
CuArray for CUD
"""
array_for_rng(::AbstractRNG, ::Type{T}, dims::Integer...) where {T} = Array{T}(undef, dims...)
array_for_rng(::CUDA.RNG, ::Type{T}, dims::Integer...) where {T} = CuArray{T}(undef, dims...)
# TODO not supported on GPU yet
array_for_rng(::CURAND.RNG, ::Type{T}, dims::Integer...) where {T} = CuArray{T}(undef, dims...)

# TODO Might want this to fail instead of fallback?
"""
    maybe_cuda
Transfers A to CUDA if A is a CuArray and issues a warning.
"""
maybe_cuda(::Any, A) = A
function maybe_cuda(::CuArray, A::AbstractArray)
    if !(A isa CuArray)
        @warn "Transferring (distribution) array to GPU, avoid overhead by transferring it once."
    end
    CuArray(A)
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

# KernelNormal

struct KernelNormal{T<:Real} <: AbstractKernelDistribution{T}
    μ::T
    σ::T
end
KernelNormal(::Type{T}=Float32) where {T} = KernelNormal{T}(0.0, 1.0)

Base.show(io::IO, dist::KernelNormal{T}) where {T} = print(io, "KernelNormal{$(T)}, μ: $(dist.μ), σ: $(dist.σ)")

function DensityInterface.logdensityof(dist::KernelNormal{T}, x) where {T}
    μ = dist.μ
    σ² = dist.σ^2
    # Unnormalized like MeasureTheroy logdensity_def
    ℓ = -T(0.5) * ((T(x) - μ)^2 / σ²)
    ℓ - log(dist.σ) - log(sqrt(T(2π)))
end

Base.rand(rng::AbstractRNG, dist::KernelNormal{T}) where {T} = dist.σ * randn(rng, T) + dist.μ

Base.maximum(::KernelNormal{T}) where {T} = typemax(T)
Base.minimum(::KernelNormal{T}) where {T} = typemin(T)
insupport(::KernelNormal, ::Real) = true
Bijectors.bijector(::KernelNormal) = Bijectors.Identity{0}()

# KernelExponential

struct KernelExponential{T<:Real} <: AbstractKernelDistribution{T}
    λ::T
end
KernelExponential(::Type{T}=Float32) where {T} = KernelExponential{T}(1.0)

Base.show(io::IO, dist::KernelExponential{T}) where {T} = print(io, "KernelExponential{$(T)}, λ: $(dist.λ)")

DensityInterface.logdensityof(dist::KernelExponential{T}, x) where {T} = insupport(dist, x) ? -dist.λ * T(x) + log(dist.λ) : -typemax(T)

Base.rand(rng::AbstractRNG, dist::KernelExponential{T}) where {T} = randexp(rng, T) / dist.λ

Base.maximum(::KernelExponential{T}) where {T} = typemax(T)
Base.minimum(::KernelExponential{T}) where {T} = zero(T)
insupport(dist::KernelExponential, x) = minimum(dist) <= x
Bijectors.bijector(::KernelExponential) = Bijectors.Log{0}()

# KernelUniform

struct KernelUniform{T<:Real} <: AbstractKernelDistribution{T}
    min::T
    max::T
end
KernelUniform(::Type{T}=Float32) where {T} = KernelUniform{T}(0.0, 1.0)

Base.show(io::IO, dist::KernelUniform{T}) where {T} = print(io, "KernelUniform{$(T)}, a: $(dist.min), b: $(dist.max)")

DensityInterface.logdensityof(dist::KernelUniform{T}, x) where {T<:Real} = insupport(dist, x) ? -log(dist.max - dist.min) : -typemax(T)

Base.rand(rng::AbstractRNG, dist::KernelUniform{T}) where {T} = (dist.max - dist.min) * rand(rng, T) + dist.min

Base.maximum(dist::KernelUniform) = dist.max
Base.minimum(dist::KernelUniform) = dist.min
insupport(dist::KernelUniform, x) = minimum(dist) <= x <= maximum(dist)
Bijectors.bijector(dist::KernelUniform) = Bijectors.TruncatedBijector{0}(minimum(dist), maximum(dist))

# KernelCircularUniform

struct KernelCircularUniform{T<:Real} <: AbstractKernelDistribution{T} end

KernelCircularUniform(::Type{T}=Float32) where {T} = KernelCircularUniform{T}()

Base.show(io::IO, ::KernelCircularUniform{T}) where {T} = print(io, "KernelCircularUniform{$(T)}")

DensityInterface.logdensityof(dist::KernelCircularUniform{T}, x) where {T} = insupport(dist, x) ? -log(T(2π)) : -typemax(T)

Base.rand(rng::AbstractRNG, ::KernelCircularUniform{T}) where {T} = T(2π) * rand(rng, T)

Base.maximum(::KernelCircularUniform{T}) where {T} = T(2π)
Base.minimum(::KernelCircularUniform{T}) where {T} = zero(T)
insupport(dist::KernelCircularUniform, x) = minimum(dist) <= x <= maximum(dist)
Bijectors.bijector(::KernelCircularUniform) = Circular{0}()

# KernelBinaryMixture

struct KernelBinaryMixture{T<:Real,U<:AbstractKernelDistribution{T},V<:AbstractKernelDistribution{T}} <: AbstractKernelDistribution{T}
    dist_1::U
    dist_2::V
    # Prefer log here, since the logdensity will be used more often than rand
    log_weight_1::T
    log_weight_2::T
    KernelBinaryMixture(dist_1::U, dist_2::V, weight_1, weight_2) where {T,U<:AbstractKernelDistribution{T},V<:AbstractKernelDistribution{T}} = new{T,U,V}(dist_1, dist_2, Float32(log(weight_1 / (weight_1 + weight_2))), Float32(log(weight_2 / (weight_1 + weight_2))))
end

Base.show(io::IO, dist::KernelBinaryMixture{T}) where {T} = print(io, "KernelBinaryMixture{$(T)}\n  components: $(dist.dist_1), $(dist.dist_2) \n  log weights: $(dist.log_weight_1), $(dist.log_weight_2)")

DensityInterface.logdensityof(dist::KernelBinaryMixture{T}, x) where {T} = insupport(dist, x) ? logaddexp(dist.log_weight_1 + logdensityof(dist.dist_1, x), dist.log_weight_2 + logdensityof(dist.dist_2, x)) : -typemax(T)

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
insupport(dist::KernelBinaryMixture, x) = minimum(dist) <= x <= maximum(dist)
Bijectors.bijector(dist::KernelBinaryMixture) = Bijectors.TruncatedBijector(minimum(dist), maximum(dist))
