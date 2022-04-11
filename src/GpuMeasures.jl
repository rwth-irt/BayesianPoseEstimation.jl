# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using CUDA
using MCMCDepth
using MeasureTheory
using LogExpFunctions
using Logging
using Random
using TransformVariables

# TODO Rename to KernelMeasures.jl
# TODO add to_gpu(d::GpuProductMeasure) to interface? Maybe with a new supertype AbstractVectorizedMeasure


# Rename to AbstractKernelMeasure
# 
"""
Measures which are isbitstype and support execution on the GPU (rand, logdensity, etc.)
Conversions:
- kernel_measure(d, ::Type{T}): Convert the MeasureTheory.jl measure to the corresponding kernel measure, T as parameter so we can provide a default (Float32)
- (optional) measure_theory(d): Convert the AbstractGpuMeasure to the corresponding MeasureTheory.jl measure
- as(d): scalar TransformVariable
Kernels, must be type stable & use isbits types:
- `rand(rng, d::MyGpuMeasure{T})::T` create a single random number
- `logdensity(d::MyGpuMeasure{T}, x)::T` evaluate the unnormalized logdensity
- `logpdf(d::MyGpuMeasure{T}, x)::T` evaluate the normalized logdensity

Most of the time Float64 precision is not required, especially for GPU computations.
Thus, we default to Float32, mostly for memory capacity reasons.
"""
abstract type AbstractKernelMeasure{T} <: AbstractMeasure end

"""
Mutate `M` with random samples from the measure `d` using rng.
"""
function Random.rand!(rng::AbstractRNG, d::AbstractKernelMeasure, M::AbstractArray{T}) where {T}
    # Broadcast rand. otherwise we would get the same value in every entry of M
    # Do not put T in a tuple, CUDA compilation will fail https://github.com/JuliaGPU/CUDA.jl/issues/261 
    M .= rand.((rng,), (d,))
end

"""
Mutate `M` with random samples from the vectorized measure `d` using rng.
"""
function Random.rand!(rng::AbstractRNG, d::AbstractArray{<:AbstractKernelMeasure}, M::AbstractArray{T}) where {T}
    # Do not put T in a tuple, CUDA compilation will fail https://github.com/JuliaGPU/CUDA.jl/issues/261 
    M .= rand.((rng,), d)
end

# Non modifying rand must create array based on RNG type
"""
Return an Array of type `T` with `dim` dimensions and `dims` dimensions sampled from the measure `d`.
"""
Base.rand(rng::AbstractRNG, d::AbstractKernelMeasure{T}, dim::Integer=1, dims::Integer...) where {T} = rand!(rng, d, Array{T}(undef, dim, dims...))

# WARN CUDA.RNG is not isbits?
#TODO GLOBAL_RNG seems the only one to be working, overriding RNG does not feel right
"""
Return a CUDA array of type `T` with `dim` dimensions and `dims` dimensions sampled from the measure `d`.
"""
Base.rand(::Union{CUDA.RNG,CURAND.RNG}, d::AbstractKernelMeasure{T}, dim::Integer=1, dims::Integer...) where {T} = rand!(Random.GLOBAL_RNG, d, CuArray{T}(undef, dim, dims...))

# Orthogonal rand / rand! are implemented by MeasureBase using GLOBAL_RNG 🙂

# KernelNormal

struct KernelNormal{T<:Real} <: AbstractKernelMeasure{T}
    μ::T
    σ::T
end

KernelNormal(::Type{T}=Float32) where {T} = KernelNormal{T}(0.0, 1.0)
KernelNormal(::Normal{()}, ::Type{T}=Float32) where {T} = KernelNormal(T)
KernelNormal(d::Normal{(:μ, :σ)}, ::Type{T}=Float32) where {T} = KernelNormal{T}(d.μ, d.σ)

kernel_measure(d::Normal, ::Type{T}=Float32) where {T} = KernelNormal(d, T)
measure_theory(d::KernelNormal) = Normal(d.μ, d.σ)
TransformVariables.as(::KernelNormal) = asℝ

Base.show(io::IO, d::KernelNormal{T}) where {T} = print(io, "GpuNormal{$(T)}, μ: $(d.μ), σ: $(d.σ)")

function MeasureTheory.logdensity(d::KernelNormal{T}, x) where {T}
    μ = d.μ
    σ² = d.σ^2
    -T(0.5) * ((T(x) - μ)^2 / σ²)
end

MeasureTheory.logpdf(d::KernelNormal{T}, x) where {T<:Real} = logdensity(d, x) - log(d.σ) - log(sqrt(T(2π)))
Base.rand(rng::AbstractRNG, d::KernelNormal{T}) where {T} = d.σ * randn(rng, T) + d.μ

# KernelExponential

struct KernelExponential{T<:Real} <: AbstractKernelMeasure{T}
    λ::T
end

KernelExponential(::Type{T}=Float32) where {T} = KernelExponential{T}(1.0)
KernelExponential(::Exponential{()}, ::Type{T}=Float32) where {T} = KernelExponential(T)
KernelExponential(d::Exponential{(:λ,)}, ::Type{T}=Float32) where {T} = KernelExponential{T}(d.λ)
KernelExponential(d::Exponential{(:β,)}, ::Type{T}=Float32) where {T} = KernelExponential{T}(1 / d.β)

kernel_measure(d::Exponential, ::Type{T}=Float32) where {T} = KernelExponential(d, T)
measure_theory(d::KernelExponential) = Exponential{(:λ,)}(d.λ)
TransformVariables.as(::KernelExponential) = asℝ₊

Base.show(io::IO, d::KernelExponential{T}) where {T} = print(io, "GpuExponential{$(T)}, λ: $(d.λ)")

MeasureTheory.logdensity(d::KernelExponential{T}, x) where {T} = -d.λ * T(x)
MeasureTheory.logpdf(d::KernelExponential, x) = logdensity(d, x) + log(d.λ)
Base.rand(rng::AbstractRNG, d::KernelExponential{T}) where {T} = randexp(rng, T) / d.λ

# KernelUniform

struct KernelUniform{T<:Real} <: AbstractKernelMeasure{T}
    a::T
    b::T
end

KernelUniform(::Type{T}=Float32) where {T} = KernelUniform{T}(0.0, 1.0)
KernelUniform(::UniformInterval{()}, ::Type{T}=Float32) where {T} = KernelUniform{T}(0.0, 1.0)
KernelUniform(d::UniformInterval{(:a, :b)}, ::Type{T}=Float32) where {T} = KernelUniform{T}(d.a, d.b)

kernel_measure(d::UniformInterval, ::Type{T}=Float32) where {T} = KernelUniform(d, T)
measure_theory(d::KernelUniform) = UniformInterval(d.a, d.b)
TransformVariables.as(d::KernelUniform) = as(Real, d.a, d.b)

Base.show(io::IO, d::KernelUniform{T}) where {T} = print(io, "GpuUniformInterval{$(T)}, a: $(d.a), b: $(d.b)")

MeasureTheory.logdensity(d::KernelUniform{T}, x) where {T<:Real} = d.a <= x <= d.b ? zero(T) : -typemax(T)
MeasureTheory.logpdf(d::KernelUniform, x) = logdensity(d, x) - log(d.b - d.a)

Base.rand(rng::AbstractRNG, d::KernelUniform{T}) where {T} = (d.b - d.a) * rand(rng, T) + d.a

# KernelCircularUniform

struct KernelCircularUniform{T<:Real} <: AbstractKernelMeasure{T} end

KernelCircularUniform(::Type{T}=Float32) where {T} = KernelCircularUniform{T}()

kernel_measure(::CircularUniform, ::Type{T}=Float32) where {T} = KernelCircularUniform(T)
measure_theory(::KernelCircularUniform) = CircularUniform()
TransformVariables.as(::KernelCircularUniform) = as○

Base.show(io::IO, ::KernelCircularUniform{T}) where {T} = print(io, "GpuCircularUniform{$(T)}")

MeasureTheory.logdensity(::KernelCircularUniform{T}, x) where {T} = logdensity(KernelUniform{T}(0, 2π), x)
MeasureTheory.logpdf(d::KernelCircularUniform{T}, x) where {T} = logdensity(d, x) - log(T(2π))

Base.rand(rng::AbstractRNG, ::KernelCircularUniform{T}) where {T} = rand(rng, KernelUniform{T}(0, 2π))

# KernelBinaryMixture

struct KernelBinaryMixture{T<:Real,U<:AbstractKernelMeasure{T},V<:AbstractKernelMeasure{T}} <: AbstractKernelMeasure{T}
    c1::U
    c2::V
    log_w1::T
    log_w2::T
    KernelBinaryMixture(c1::U, c2::V, w1, w2) where {T,U<:AbstractKernelMeasure{T},V<:AbstractKernelMeasure{T}} = new{T,U,V}(c1, c2, Float32(log(w1 / (w1 + w2))), Float32(log(w2 / (w1 + w2))))
end

KernelBinaryMixture(d::BinaryMixture, ::Type{T}=Float32) where {T} = KernelBinaryMixture(kernel_measure(d.c1, T), kernel_measure(d.c2, T), exp(d.log_w1), exp(d.log_w2))

kernel_measure(d::BinaryMixture, ::Type{T}=Float32) where {T} = KernelBinaryMixture(d, T)
measure_theory(d::KernelBinaryMixture) = BinaryMixture(measure_theory(d.c1), measure_theory(d.c2), exp(d.log_w1), exp(d.log_w2))
# TODO Does it make sense? Support of rand is the union of c1 & c2, support of logdensity is the intersection. So probably only makes sense for same supports or everything.
function TransformVariables.as(d::KernelBinaryMixture)
    as(d.c1) == as(d.c2) ? as(d.c1) : asℝ
end

Base.show(io::IO, d::KernelBinaryMixture{T}) where {T} = print(io, "GpuBinaryMixture{$(T)}\n  components: $(d.c1), $(d.c2) \n  log weights: $(d.log_w1), $(d.log_w2)")

MeasureTheory.logdensity(d::KernelBinaryMixture, x) = logaddexp(d.log_w1 + logpdf(d.c1, x), d.log_w2 + logpdf(d.c2, x))
MeasureTheory.logpdf(d::KernelBinaryMixture, x) = logdensity(d, x)

function Base.rand(rng::AbstractRNG, d::KernelBinaryMixture{T}) where {T}
    log_u = log(rand(rng, T))
    if log_u < d.log_w1
        rand(rng, d.c1)
    else
        rand(rng, d.c2)
    end
end


# AbstractVectorizedMeasure

"""
Vectorization support by storing multiple measures in an array for broadcasting.

Implement the AbstractKernelMeasure interfaces and additionally:
- MeasureTheory.marginals():    Return the internal array of measures

You can use:
- as
- rand & rand!
- maybe_to_gpu(d, M)
- vectorized_logdensity(d, M)
- vectorized_logpdf(d, M)
- to_cpu(d)
- to_gpu(d)
"""
abstract type AbstractVectorizedMeasure{T} <: AbstractKernelMeasure{T} end

"""
    to_cpu(d)
Transfer the internal measures to the CPU.
"""
to_cpu(d::T) where {T<:AbstractVectorizedMeasure} = T.name.wrapper(Array(marginals(d)))

"""
    to_gpu(d)
Transfer the internal measures to the GPU.
"""
to_gpu(d::T) where {T<:AbstractVectorizedMeasure} = T.name.wrapper(CuArray(marginals(d)))

"""
    maybe_to_gpu(d, M)
Transfer the the marginals of the measure to the GPU if M is a CuArray.
Prevents CUDA kernel compilation errors.
"""
function maybe_to_gpu(d::AbstractVectorizedMeasure, M)
    if (M isa CuArray) && !(marginals(d) isa CuArray)
        @warn "Transferring vectorized measure to GPU, avoid overhead by calling d=to_gpu(d::AbstractVectorizedMeasure) once."
        return to_gpu(d)
    else
        return d
    end
end

# Let the broadcasting magic do its work on the internal measures
function Random.rand!(rng::AbstractRNG, d::AbstractVectorizedMeasure, M::AbstractArray)
    d = maybe_to_gpu(d, M)
    rand!(rng, marginals(d), M)
end

# Automatically choose correct size for the measure
Base.rand(rng::AbstractRNG, d::AbstractVectorizedMeasure) = rand(rng, d, size(marginals(d))...)
# Resolve ambiguity for specific RNG
Base.rand(rng::Union{CUDA.RNG,CURAND.RNG}, d::AbstractVectorizedMeasure) = rand(rng, d, size(marginals(d))...)

"""
    broadcast_logdensity(d, M)
Broadcasts the logdensity function, takes care of transferring the measure to the GPU if required.
"""
function broadcast_logdensity(d::AbstractVectorizedMeasure, M)
    d = maybe_to_gpu(d, M)
    # TODO lazy broadcasted?
    logdensity.(marginals(d), M)
end

"""
    broadcast_logpdf(d, M)
Broadcasts the logpdf function, takes care of transferring the measure to the GPU if required.
"""
function broadcast_logpdf(d::AbstractVectorizedMeasure, M)
    d = maybe_to_gpu(d, M)
    logpdf.(marginals(d), M)
end

"""
    as(d)
Scalar transform variable for the marginals
"""
TransformVariables.as(d::AbstractVectorizedMeasure) = marginals(d) |> first |> as

# KernelProduct

"""
    KernelProduct
Assumes independent marginals, whose logdensity is the sum of each individual logdensity (like the MeasureTheory Product measure).
"""
struct KernelProduct{T<:Real,U<:AbstractKernelMeasure{T},V<:AbstractArray{U}} <: AbstractVectorizedMeasure{T}
    # WARN CUDA kernels only work for the same Measure with different parametrization
    marginals::V
end

"""
    KernelProduct(d,T)
Convert a MeasureTheory ProductMeasure into a gpu measure.
Warning: The marginals of the measure must be of the same type to transfer them to an CuArray.
"""
KernelProduct(d::ProductMeasure, ::Type{T}=Float32) where {T} = kernel_measure.(marginals(d), (T,)) |> KernelProduct

kernel_measure(d::ProductMeasure, ::Type{T}=Float32) where {T} = KernelProduct(d, T)
measure_theory(d::KernelProduct) = MeasureBase.productmeasure(identity, d.marginals |> Array .|> measure_theory)

MeasureTheory.marginals(d::KernelProduct) = d.marginals

Base.show(io::IO, d::KernelProduct{T}) where {T} = print(io, "GpuProductMeasure{$(T)}\n  measures: $(typeof(d.marginals))\n  size: $(size(d.marginals))")

function MeasureTheory.logdensity(d::KernelProduct, x)
    ℓ = broadcast_logdensity(d, x)
    sum(ℓ)
end
function MeasureTheory.logpdf(d::KernelProduct, x)
    ℓ = broadcast_logpdf(d, x)
    sum(ℓ)
end

# VectorizedMeasure

"""
    VectorizedMeasure
Behaves similar to the ProductMeasure but assumes a vectorization of the data over last dimension.
"""
struct VectorizedMeasure{T<:Real,U<:AbstractKernelMeasure{T},V<:AbstractArray{U}} <: AbstractVectorizedMeasure{T}
    marginals::V
end

"""
    VectorizedMeasure(d, T)
Convert a MeasureTheory ProductMeasure into a gpu measure.
Warning: The marginals of the measure must be of the same type to transfer them to an CuArray.
"""
VectorizedMeasure(d::ProductMeasure, ::Type{T}=Float32) where {T} = kernel_measure(d, T) |> marginals |> VectorizedMeasure

MeasureTheory.marginals(d::VectorizedMeasure) = d.marginals

Base.show(io::IO, d::VectorizedMeasure{T}) where {T} = print(io, "GpuVectorizedMeasure{$(T)}\n  internal: $(eltype(d.marginals)) \n  size: $(size(d.marginals))")

Base.size(d::VectorizedMeasure) = d.size

# TODO find a better place
"""
    reduce_to_last_dim(op, M)
Reduces all dimensions but the last one.
"""
function reduce_to_last_dim(op, M::AbstractArray{<:Any,N}) where {N}
    R = reduce(op, M; dims=(1:N-1...,))
    dropdims(R; dims=(1:N-1...,))
end

function MeasureTheory.logdensity(d::VectorizedMeasure, x)
    ℓ = broadcast_logdensity(d, x)
    reduce_to_last_dim(+, ℓ)
end
function MeasureTheory.logpdf(d::VectorizedMeasure, x)
    ℓ = broadcast_logpdf(d, x)
    reduce_to_last_dim(+, ℓ)
end
