# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using CUDA
using DensityInterface
using MeasureTheory
using MCMCDepth
using LogExpFunctions
using Logging
using Random
using TransformVariables

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

The Interface requires the following to be implemented:
- as(d): scalar TransformVariable
- `Base.rand(rng, d::MyKernelDistribution{T})::T` generate a single random number from the distribution
- `logpdf(d::MyKernelDistribution{T}, x)::T` evaluate the normalized logdensity

Conversion functions used to convert Product measures and testing:
- kernel_distribution(d, ::Type{T}): Convert the MeasureTheory.jl measure to the corresponding kernel distribution, T as parameter so we can provide a default (Float32)
- (optional) measure_theory(d): Convert the KernelDistribution to the corresponding MeasureTheory.jl measure

Most of the time Float64 precision is not required, especially for GPU computations.
Thus, we default to Float32, mostly for memory capacity reasons.
"""
abstract type AbstractKernelDistribution{T} end
const KernelOrKernelArray{T} = Union{AbstractKernelDistribution{T},AbstractArray{<:AbstractKernelDistribution{T}}}

# DensityInterface

@inline DensityInterface.DensityKind(::AbstractKernelDistribution) = IsDensity()
"""
    logdensityof(d, x)
Implement DensityInterface, by providing the normalized logdensity of the distribution.
Uses broadcasting for arrays.
"""
DensityInterface.logdensityof(d::AbstractKernelDistribution, x) = _logdensity_of(d, x)

# Avoid ambiguities by not specializing x for logdensityof
_logdensity_of(d::AbstractKernelDistribution, x) = logpdf(d, x)
_logdensity_of(d::AbstractKernelDistribution, x::AbstractArray) = logpdf.((d,), x)
function _logdensityof(D::AbstractArray{<:AbstractKernelDistribution}, x)
    D = maybe_cuda(x, D)
    logpdf.(D, x)
end

# Random interface
# Quite elaborate to use CUDA.RNG inside the kernel

"""
    rand!(rng, d, A)
Mutate the array A by sampling from the distribution `d`.
"""
Random.rand!(rng::AbstractRNG, d::KernelOrKernelArray, A::AbstractArray) = _rand!(rng, d, A)
"""
    rand(rng, m, dims)
Sample an Array from the distribution `d` of size `dims`.
"""
function Base.rand(rng::AbstractRNG, d::KernelOrKernelArray{T}, dims::Integer...) where {T}
    A = array_for_rng(rng, T, dims...)
    rand!(rng, d, A)
end

"""
    rand(rng, m)
Sample an Array from the distribution `d` of size 1.
"""
Base.rand(rng::AbstractRNG, d::AbstractKernelDistribution) = rand(rng, d, 1)[]

# Orthogonal methods
Random.rand!(d::AbstractKernelDistribution, A::AbstractArray) = rand!(Random.GLOBAL_RNG, d, A)
Base.rand(d::AbstractKernelDistribution, dims::Integer...) = rand(Random.GLOBAL_RNG, d, dims...)
Base.rand(d::AbstractKernelDistribution) = rand(Random.GLOBAL_RNG, d)


# CPU implementation

_rand!(rng::AbstractRNG, d, A::Array) = cpu_rand!(rng, d, A)
function cpu_rand!(rng::AbstractRNG, d::AbstractKernelDistribution, A::Array)
    A .= rand.((rng,), (d,))
end
function cpu_rand!(rng::AbstractRNG, D::AbstractArray{<:AbstractKernelDistribution}, A::Array)
    A .= rand.((rng,), D)
end

# GPU implementation

function _rand!(rng::CUDA.RNG, d::KernelOrKernelArray, A::CuArray)
    # WARN any RNG other than CUDA.RNG must run on the CPU and copy the data to the GPU → SLOW
    d = maybe_cuda(A, d)
    gpu_rand!(d, A, rng.seed, rng.counter)
    new_counter = Int64(rng.counter) + length(A)
    overflow, remainder = fldmod(new_counter, typemax(UInt32))
    rng.seed += overflow
    rng.counter = remainder
    return A
end

# Function barrier for CUDA.RNG.seed / counter
# Wrapping rng in Tuple for broadcasting does not work → anonymous function is the workaround 
# Thanks vchuravy https://github.com/JuliaGPU/CUDA.jl/issues/1480#issuecomment-1102245813
function gpu_rand!(d::AbstractKernelDistribution, A::CuArray, seed, counter)
    A .= (x -> rand(device_rng(seed, counter), x)).((d,))
end
function gpu_rand!(D::CuArray{<:AbstractKernelDistribution}, A::CuArray, seed, counter)
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
# TODO not supported in device_rng yet
# array_for_rng(::CURAND.RNG, ::Type{T}, dims::Integer...) where {T} = CuArray{T}(undef, dims...)

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
KernelNormal(::Normal{()}, T::Type=Float32) = KernelNormal(T)
KernelNormal(d::Normal{(:μ, :σ)}, T::Type=Float32) = KernelNormal{T}(d.μ, d.σ)

kernel_distribution(d::Normal, T::Type=Float32) = KernelNormal(d, T)
measure_theory(d::KernelNormal) = Normal(d.μ, d.σ)
TransformVariables.as(::KernelNormal) = asℝ

Base.show(io::IO, d::KernelNormal{T}) where {T} = print(io, "KernelNormal{$(T)}, μ: $(d.μ), σ: $(d.σ)")

function logpdf(d::KernelNormal{T}, x) where {T}
    μ = d.μ
    σ² = d.σ^2
    # Unnormalized like MeasureTheroy logdensity_def
    ℓ = -T(0.5) * ((T(x) - μ)^2 / σ²)
    ℓ - log(d.σ) - log(sqrt(T(2π)))
end

Base.rand(rng::AbstractRNG, d::KernelNormal{T}) where {T} = d.σ * randn(rng, T) + d.μ

# KernelExponential

struct KernelExponential{T<:Real} <: AbstractKernelDistribution{T}
    λ::T
end

KernelExponential(::Type{T}=Float32) where {T} = KernelExponential{T}(1.0)
KernelExponential(::Exponential{()}, T::Type=Float32) = KernelExponential(T)
KernelExponential(d::Exponential{(:λ,)}, ::Type{T}=Float32) where {T} = KernelExponential{T}(d.λ)
KernelExponential(d::Exponential{(:β,)}, ::Type{T}=Float32) where {T} = KernelExponential{T}(1 / d.β)

kernel_distribution(d::Exponential, T::Type=Float32) = KernelExponential(d, T)
measure_theory(d::KernelExponential) = Exponential{(:λ,)}(d.λ)
TransformVariables.as(::KernelExponential) = asℝ₊

Base.show(io::IO, d::KernelExponential{T}) where {T} = print(io, "KernelExponential{$(T)}, λ: $(d.λ)")

logpdf(d::KernelExponential{T}, x) where {T} = x >= 0 ? -d.λ * T(x) + log(d.λ) : -typemax(T)

Base.rand(rng::AbstractRNG, d::KernelExponential{T}) where {T} = randexp(rng, T) / d.λ

# KernelUniform

struct KernelUniform{T<:Real} <: AbstractKernelDistribution{T}
    a::T
    b::T
end

KernelUniform(::Type{T}=Float32) where {T} = KernelUniform{T}(0.0, 1.0)
KernelUniform(::UniformInterval{()}, ::Type{T}=Float32) where {T} = KernelUniform{T}(0.0, 1.0)
KernelUniform(d::UniformInterval{(:a, :b)}, ::Type{T}=Float32) where {T} = KernelUniform{T}(d.a, d.b)

kernel_distribution(d::UniformInterval, T::Type=Float32) = KernelUniform(d, T)
measure_theory(d::KernelUniform) = UniformInterval(d.a, d.b)
TransformVariables.as(d::KernelUniform) = as(Real, d.a, d.b)

Base.show(io::IO, d::KernelUniform{T}) where {T} = print(io, "KernelUniform{$(T)}, a: $(d.a), b: $(d.b)")

logpdf(d::KernelUniform{T}, x) where {T<:Real} = d.a <= x <= d.b ? -log(d.b - d.a) : -typemax(T)

Base.rand(rng::AbstractRNG, d::KernelUniform{T}) where {T} = (d.b - d.a) * rand(rng, T) + d.a

# KernelCircularUniform

struct KernelCircularUniform{T<:Real} <: AbstractKernelDistribution{T} end

KernelCircularUniform(::Type{T}=Float32) where {T} = KernelCircularUniform{T}()

kernel_distribution(::CircularUniform, T::Type=Float32) = KernelCircularUniform(T)
measure_theory(::KernelCircularUniform) = CircularUniform()
TransformVariables.as(::KernelCircularUniform) = as○

Base.show(io::IO, ::KernelCircularUniform{T}) where {T} = print(io, "KernelCircularUniform{$(T)}")

logpdf(::KernelCircularUniform{T}, x) where {T} = 0 <= x <= 2π ? -log(T(2π)) : -typemax(T)

Base.rand(rng::AbstractRNG, ::KernelCircularUniform{T}) where {T} = T(2π) * rand(rng, T)

# KernelBinaryMixture

struct KernelBinaryMixture{T<:Real,U<:AbstractKernelDistribution{T},V<:AbstractKernelDistribution{T}} <: AbstractKernelDistribution{T}
    c1::U
    c2::V
    # Prefer log here, since the logdensity will be used more often than rand
    log_w1::T
    log_w2::T
    KernelBinaryMixture(c1::U, c2::V, w1, w2) where {T,U<:AbstractKernelDistribution{T},V<:AbstractKernelDistribution{T}} = new{T,U,V}(c1, c2, Float32(log(w1 / (w1 + w2))), Float32(log(w2 / (w1 + w2))))
end

KernelBinaryMixture(d::BinaryMixture, T::Type=Float32) = KernelBinaryMixture(kernel_distribution(d.c1, T), kernel_distribution(d.c2, T), exp(d.log_w1), exp(d.log_w2))

kernel_distribution(d::BinaryMixture, T::Type=Float32) = KernelBinaryMixture(d, T)
measure_theory(d::KernelBinaryMixture) = BinaryMixture(measure_theory(d.c1), measure_theory(d.c2), exp(d.log_w1), exp(d.log_w2))

# TODO Support of rand is the union of c1 & c2, support of logdensity is the intersection. So probably only makes sense for same supports or everything.
function TransformVariables.as(d::KernelBinaryMixture)
    as(d.c1) == as(d.c2) ? as(d.c1) : asℝ
end

Base.show(io::IO, d::KernelBinaryMixture{T}) where {T} = print(io, "KernelBinaryMixture{$(T)}\n  components: $(d.c1), $(d.c2) \n  log weights: $(d.log_w1), $(d.log_w2)")

logpdf(d::KernelBinaryMixture, x) = logaddexp(d.log_w1 + logpdf(d.c1, x), d.log_w2 + logpdf(d.c2, x))

function Base.rand(rng::AbstractRNG, d::KernelBinaryMixture{T}) where {T}
    log_u = log(rand(rng, T))
    if log_u < d.log_w1
        rand(rng, d.c1)
    else
        rand(rng, d.c2)
    end
end

# TODO move to separate file.
# AbstractVectorizedKernel

"""
Vectorization support by storing multiple measures in an array for broadcasting.

Implement:
- MeasureTheory.marginals(): Return the internal array of measures
- DensityInterface.logdensityof(d::AbstractVectorizedKernel, x)

You can use:
- as
- rand & rand!
- to_cpu(d)
- to_gpu(d)
"""
abstract type AbstractVectorizedKernel{T} <: AbstractKernelDistribution{T} end

"""
    to_cpu(d)
Transfer the internal distributions to the CPU.
"""
to_cpu(d::T) where {T<:AbstractVectorizedKernel} = T.name.wrapper(Array(marginals(d)))

"""
    to_gpu(d)
Transfer the internal distributions to the GPU.
"""
to_gpu(d::T) where {T<:AbstractVectorizedKernel} = T.name.wrapper(CuArray(marginals(d)))

"""
    rand(rng, m, dims)
Mutate the array `A` by sampling from the distribution `d`.
"""
Random.rand!(rng::AbstractRNG, d::AbstractVectorizedKernel, A::AbstractArray) = rand!(rng, marginals(d), A)

"""
    rand(rng, m, dims)
Sample an array from the distribution `d` of size `dims`.
"""
Base.rand(rng::AbstractRNG, d::AbstractVectorizedKernel, dims::Integer...) = rand(rng, marginals(d), size(marginals(d))..., dims...)

"""
    rand(rng, m, dims)
Sample an array from the distribution `d` of size 1.
"""
Base.rand(rng::AbstractRNG, d::AbstractVectorizedKernel) = rand(rng, marginals(d), size(marginals(d))..., 1)

"""
    as(d)
Scalar transform variable for the marginals
"""
TransformVariables.as(d::AbstractVectorizedKernel) = marginals(d) |> first |> as

"""
    measure_theory(d)
Converts the vectorized kernel distribution to a product measure.
"""
measure_theory(d::AbstractVectorizedKernel) = MeasureBase.productmeasure(marginals(d) |> Array .|> measure_theory)

# KernelProduct

"""
    KernelProduct
Assumes independent marginals, whose logdensity is the sum of each individual logdensity (like the MeasureTheory Product measure).
"""
struct KernelProduct{T<:Real,U<:AbstractKernelDistribution{T},V<:AbstractArray{U}} <: AbstractVectorizedKernel{T}
    # WARN CUDA kernels only work for the same Measure with different parametrization
    marginals::V
end

"""
    KernelProduct(d,T)
Convert a MeasureTheory AbstractProductMeasure to a KernelProduct.
Warning: The marginals of the measure must be of the same type to transfer them to an CuArray.
"""
KernelProduct(d::AbstractProductMeasure, T::Type=Float32) = kernel_distribution.(marginals(d), (T,)) |> KernelProduct

"""
    KernelProduct(d)
Convert an AbstractVectorizedKernel to a KernelVectorized.
"""
KernelProduct(d::AbstractVectorizedKernel) = KernelProduct(marginals(d))

MeasureTheory.marginals(d::KernelProduct) = d.marginals

Base.show(io::IO, d::KernelProduct{T}) where {T} = print(io, "KernelProduct{$(T)}\n  marginals: $(typeof(d.marginals))\n  size: $(size(d.marginals))")

function DensityInterface.logdensityof(d::KernelProduct, x)
    ℓ = _logdensityof(d.marginals, x)
    sum(ℓ)
end

# KernelVectorized

"""
    KernelVectorized
Behaves similar to the ProductKernel but assumes a vectorization of the data over last dimension.
"""
struct KernelVectorized{T<:Real,U<:AbstractKernelDistribution{T},V<:AbstractArray{U}} <: AbstractVectorizedKernel{T}
    marginals::V
end

"""
    KernelVectorized(d)
Convert an AbstractVectorizedKernel to a KernelVectorized.
"""
KernelVectorized(d::AbstractVectorizedKernel) = KernelVectorized(marginals(d))

"""
    KernelVectorized(d)
Convert a KernelProduct to a KernelVectorized.
"""
KernelVectorized(d::KernelProduct, T::Type=Float32) = kernel_distribution.(marginals(d), (T,)) |> KernelVectorized

"""
    KernelVectorized(d)
Convert a AbstractProductMeasure to a KernelVectorized.
"""
KernelVectorized(d::AbstractProductMeasure, T::Type=Float32) = kernel_distribution.(marginals(d), (T,)) |> KernelVectorized

"""
    kernel_distribution(d, T)
Default is a KernelVectorized which reduces to KernelProduct in case the size of the marginals and the data matches.
"""
kernel_distribution(d::AbstractProductMeasure, T::Type=Float32) = KernelVectorized(d, T)

MeasureTheory.marginals(d::KernelVectorized) = d.marginals

Base.show(io::IO, d::KernelVectorized{T}) where {T} = print(io, "KernelVectorized{$(T)}\n  marginals: $(eltype(d.marginals)) \n  size: $(size(d.marginals))")

Base.size(d::KernelVectorized) = d.size

"""
    reduce_vectorized(op, d, A)
Reduces the first `ndims(d)` dimensions of the Matrix `A` using the operator `op`. 
"""
function reduce_vectorized(op, d::KernelVectorized, A::AbstractArray)
    n_red = ndims(marginals(d))
    R = reduce(op, A; dims=(1:n_red...,))
    dropdims(R; dims=(1:n_red...,))
end

function DensityInterface.logdensityof(d::KernelVectorized, x)
    ℓ = _logdensityof(d.marginals, x)
    reduce_vectorized(+, d, ℓ)
end
