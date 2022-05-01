# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using CUDA
using DensityInterface
using MCMCDepth
using LogExpFunctions
using Logging
using Random
using TransformVariables

# TODO At one point most of the distributions could be replaced with Distributions.jl. Mixtures could be problematic as well as the transition from TransformVariables.jl to Bijectors.jl
# TODO should open a pull request to fix type of https://github.com/JuliaStats/Distributions.jl/blob/d19ac4526bab2584a84323eea4af92805f99f034/src/univariate/continuous/uniform.jl#L120

"""
MeasureTheory.jl is what I have used because of the nicer interface until now, but all the type are not isbits and can not be used on the GPU.
Distributions.jl is pretty close but not perfect for the execution on the GPU:
- Mostly type stable
- Mixtures a quirky
- Uniform is not strongly typed resulting in Float64 calculations all the time.

Here, I provide stripped-down Distributions which are isbitstype, strongly typed and thus support execution on the GPU.
KernelDistributions offer the following interface functions:
- `DensityInterface.logdensityof(d::KernelDistribution, x)`
- `Random.rand!(rng, d::KernelDistribution, A)`
- `Base.rand(rng, d::KernelDistribution, dims...)`
- maximum(d), minimum(d), insupport(d): Determine the support of the distribution

The Interface requires the following to be implemented:
- as(d): scalar TransformVariable
- `Base.rand(rng, d::MyKernelDistribution{T})::T` generate a single random number from the distribution
- `logpdf(d::MyKernelDistribution{T}, x)::T` evaluate the normalized logdensity

Most of the time Float64 precision is not required, especially for GPU computations.
Thus, we default to Float32, mostly for memory capacity reasons.
"""
abstract type AbstractKernelDistribution{T} end

# WARN parametric alias causes method ambiguities, since the parametric type is always present
const KernelOrKernelArray = Union{AbstractKernelDistribution,AbstractArray{<:AbstractKernelDistribution}}

# DensityInterface

@inline DensityInterface.DensityKind(::AbstractKernelDistribution) = IsDensity()

# KernelOrKernelArray should have the same behavior interface
"""
    logdensityof(d, x)
Implement DensityInterface, by providing the normalized logdensity of the distribution.
Uses broadcasting for arrays.
"""
DensityInterface.logdensityof(d::KernelOrKernelArray, x) = _logdensityof(d, x)

# KernelOrKernelArray in KernelDistributionsVariables.jl would cause ambiguities
_logdensityof(d::AbstractKernelDistribution, x) = logpdf(d, x)
_logdensityof(d::AbstractKernelDistribution, x::AbstractArray) = logpdf.((d,), x)

function _logdensityof(D::AbstractArray{<:AbstractKernelDistribution}, x)
    D = maybe_cuda(x, D)
    logpdf.(D, x)
end

# Random interface

"""
    rand!(rng, d, A)
Mutate the array A by sampling from the distribution `d`.
"""
Random.rand!(rng::AbstractRNG, d::KernelOrKernelArray, A::AbstractArray) = _rand!(rng, d, A)

"""
    rand(rng, d, dims)
Sample an Array from the distribution `d` of size `dims`.
"""
function Base.rand(rng::AbstractRNG, d::AbstractKernelDistribution{T}, dims::Integer...) where {T}
    A = array_for_rng(rng, T, dims...)
    rand!(rng, d, A)
end

"""
    rand(rng, d, dims)
Sample an Array from the distribution `d` of size `dims`.
"""
function Base.rand(rng::AbstractRNG, d::AbstractArray{<:AbstractKernelDistribution{T}}, dims::Integer...) where {T}
    A = array_for_rng(rng, T, dims...)
    rand!(rng, d, A)
end

"""
    rand(rng, d)
Sample an Array from the distribution `d` of size 1.
"""
Base.rand(rng::AbstractRNG, d::AbstractKernelDistribution) = rand(rng, d, 1)[]

"""
    rand(rng, d)
Sample an Array from the array of distributions `d` with the size of d.
"""
Base.rand(rng::AbstractRNG, d::AbstractArray{<:AbstractKernelDistribution{T}}) where {T} = rand(rng, d, size(d)...)

# Orthogonal methods
Random.rand!(d::AbstractKernelDistribution, A::AbstractArray) = rand!(Random.GLOBAL_RNG, d, A)
Base.rand(d::AbstractKernelDistribution, dims::Integer...) = rand(Random.GLOBAL_RNG, d, dims...)
Base.rand(d::AbstractKernelDistribution) = rand(Random.GLOBAL_RNG, d)

# Transform variables for arrays
TransformVariables.as(A::AbstractArray{<:AbstractKernelDistribution}) = as(first(A))

# CPU implementation

# _rand!(rng::AbstractRNG, d, A::Array) = cpu_rand!(rng, d, A)

function _rand!(rng::AbstractRNG, d::AbstractKernelDistribution, A::Array)
    A .= rand.((rng,), (d,))
end

function _rand!(rng::AbstractRNG, D::AbstractArray{<:AbstractKernelDistribution}, A::Array)
    A .= rand.((rng,), D)
end

# GPU implementation

function _rand!(rng::AbstractRNG, D::KernelOrKernelArray, A::CuArray{T}) where {T}
    @warn "Using unsupported RNG of type $(typeof(rng)) on CuArray. Falling back to SLOW sampling on CPU and copying it to the GPU."
    copyto!(A, rand!(rng, D, Array{T}(undef, size(A)...)))
end

function _rand!(rng::CUDA.RNG, d::KernelOrKernelArray, A::CuArray)
    d = maybe_cuda(A, d)
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
function rand_barrier(d::AbstractKernelDistribution, A::CuArray, seed, counter)
    A .= (x -> rand(device_rng(seed, counter), x)).((d,))
end

function rand_barrier(D::CuArray{<:AbstractKernelDistribution}, A::CuArray, seed, counter)
    A .= (x -> rand(device_rng(seed, counter), x)).(D)
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

Base.show(io::IO, d::KernelNormal{T}) where {T} = print(io, "KernelNormal{$(T)}, μ: $(d.μ), σ: $(d.σ)")

function logpdf(d::KernelNormal{T}, x) where {T}
    μ = d.μ
    σ² = d.σ^2
    # Unnormalized like MeasureTheroy logdensity_def
    ℓ = -T(0.5) * ((T(x) - μ)^2 / σ²)
    ℓ - log(d.σ) - log(sqrt(T(2π)))
end

Base.rand(rng::AbstractRNG, d::KernelNormal{T}) where {T} = d.σ * randn(rng, T) + d.μ

Base.maximum(::KernelNormal{T}) where {T} = typemax(T)
Base.minimum(::KernelNormal{T}) where {T} = typemin(T)
insupport(::KernelNormal, ::Real) = true
TransformVariables.as(::KernelNormal) = asℝ

# KernelExponential

struct KernelExponential{T<:Real} <: AbstractKernelDistribution{T}
    λ::T
end
KernelExponential(::Type{T}=Float32) where {T} = KernelExponential{T}(1.0)

Base.show(io::IO, d::KernelExponential{T}) where {T} = print(io, "KernelExponential{$(T)}, λ: $(d.λ)")

logpdf(d::KernelExponential{T}, x) where {T} = insupport(d, x) ? -d.λ * T(x) + log(d.λ) : -typemax(T)

Base.rand(rng::AbstractRNG, d::KernelExponential{T}) where {T} = randexp(rng, T) / d.λ

Base.maximum(::KernelExponential{T}) where {T} = typemax(T)
Base.minimum(::KernelExponential{T}) where {T} = zero(T)
insupport(d::KernelExponential, x) = minimum(d) <= x
TransformVariables.as(::KernelExponential) = asℝ₊

# KernelUniform

struct KernelUniform{T<:Real} <: AbstractKernelDistribution{T}
    a::T
    b::T
end
KernelUniform(::Type{T}=Float32) where {T} = KernelUniform{T}(0.0, 1.0)

Base.show(io::IO, d::KernelUniform{T}) where {T} = print(io, "KernelUniform{$(T)}, a: $(d.a), b: $(d.b)")

logpdf(d::KernelUniform{T}, x) where {T<:Real} = insupport(d, x) ? -log(d.b - d.a) : -typemax(T)

Base.rand(rng::AbstractRNG, d::KernelUniform{T}) where {T} = (d.b - d.a) * rand(rng, T) + d.a

Base.maximum(d::KernelUniform) = d.b
Base.minimum(d::KernelUniform) = d.a
insupport(d::KernelUniform, x) = minimum(d) <= x <= maximum(d)
TransformVariables.as(d::KernelUniform) = as(Real, d.a, d.b)

# KernelCircularUniform

struct KernelCircularUniform{T<:Real} <: AbstractKernelDistribution{T} end

KernelCircularUniform(::Type{T}=Float32) where {T} = KernelCircularUniform{T}()

Base.show(io::IO, ::KernelCircularUniform{T}) where {T} = print(io, "KernelCircularUniform{$(T)}")

logpdf(d::KernelCircularUniform{T}, x) where {T} = insupport(d, x) ? -log(T(2π)) : -typemax(T)

Base.rand(rng::AbstractRNG, ::KernelCircularUniform{T}) where {T} = T(2π) * rand(rng, T)

Base.maximum(::KernelCircularUniform{T}) where {T} = T(2π)
Base.minimum(::KernelCircularUniform{T}) where {T} = zero(T)
insupport(d::KernelCircularUniform, x) = minimum(d) <= x <= maximum(d)
TransformVariables.as(::KernelCircularUniform) = as○

# KernelBinaryMixture

struct KernelBinaryMixture{T<:Real,U<:AbstractKernelDistribution{T},V<:AbstractKernelDistribution{T}} <: AbstractKernelDistribution{T}
    c1::U
    c2::V
    # Prefer log here, since the logdensity will be used more often than rand
    log_w1::T
    log_w2::T
    KernelBinaryMixture(c1::U, c2::V, w1, w2) where {T,U<:AbstractKernelDistribution{T},V<:AbstractKernelDistribution{T}} = new{T,U,V}(c1, c2, Float32(log(w1 / (w1 + w2))), Float32(log(w2 / (w1 + w2))))
end

Base.show(io::IO, d::KernelBinaryMixture{T}) where {T} = print(io, "KernelBinaryMixture{$(T)}\n  components: $(d.c1), $(d.c2) \n  log weights: $(d.log_w1), $(d.log_w2)")

# logpdf(d::KernelBinaryMixture, x) = logpdf(d.c1, x)
# logpdf(d::KernelBinaryMixture, x) = logaddexp(d.log_w1 + logpdf(d.c1, x), d.log_w2 + logpdf(d.c2, x))
logpdf(d::KernelBinaryMixture{T}, x) where {T} = insupport(d, x) ? logaddexp(d.log_w1 + logpdf(d.c1, x), d.log_w2 + logpdf(d.c2, x)) : -typemax(T)

function Base.rand(rng::AbstractRNG, d::KernelBinaryMixture{T}) where {T}
    log_u = log(rand(rng, T))
    if log_u < d.log_w1
        rand(rng, d.c1)
    else
        rand(rng, d.c2)
    end
end

# The support of a mixture is the union of the support of its components
Base.maximum(d::KernelBinaryMixture) = max(maximum(d.c1), maximum(d.c2))
Base.minimum(d::KernelBinaryMixture) = min(minimum(d.c1), minimum(d.c2))
insupport(d::KernelBinaryMixture, x) = minimum(d) <= x <= maximum(d)
to∞(x) = isinf(x) ? TransformVariables.Infinity{x > 0}() : x
TransformVariables.as(d::KernelBinaryMixture) = as(Real, to∞(minimum(d)), to∞(maximum(d)))
