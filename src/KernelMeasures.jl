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

# TODO We would not need to separate rand_fn & transform_rand, if broadcasting `M .= rand.((rng,))` would work. Right now, only GLOBAL_RNG works with broadcasting, which defeats the purpose of providing an RNG.

# Rename to AbstractKernelMeasure
# 
"""
Measures which are isbitstype and support execution on the GPU (rand, logdensity, etc.)
Conversions:
- kernel_measure(d, ::Type{T}): Convert the MeasureTheory.jl measure to the corresponding kernel measure, T as parameter so we can provide a default (Float32)
- (optional) measure_theory(d): Convert the KernelMeasure to the corresponding MeasureTheory.jl measure
- as(d): scalar TransformVariable
Random kernels are separated from the random number generation to allow broadcasting.
They must be type stable & use isbits types.
- `rand_fn(::Type{<:MyKernelMeasure})` return the primitive random number generator, e.g. randn
- `transform_rand(d::MyKernelMeasure, x::Real)` transform the value x provided by rand_fn
- `logdensity(d::MyKernelMeasure{T}, x)::T` evaluate the unnormalized logdensity
- `logpdf(d::MyKernelMeasure{T}, x)::T` evaluate the normalized logdensity

Most of the time Float64 precision is not required, especially for GPU computations.
Thus, we default to Float32, mostly for memory capacity reasons.
"""
abstract type AbstractKernelMeasure{T} <: AbstractMeasure end

"""
    maybe_cuda
Transfers B to CUDA if A is a CuArray and issues a warning.
"""
maybe_cuda(::Any, B) = B
function maybe_cuda(::Type{<:CuArray}, B)
    if !(B isa CuArray)
        @warn "Transferring measure array to GPU, avoid overhead by transferring it once."
    end
    CuArray(B)
end

"""
    broadcast_transform_rand(m, A)
Broadcasts the transform_rand for the measure `m` over the array `A` which contains the primitive random numbers.
"""
function broadcast_transform_rand(m::AbstractKernelMeasure, A::AbstractArray)
    transform_rand.((m,), A)
end

"""
    broadcast_transform_rand(m, A)
Broadcasts the transform_rand for the measure `m` over the array `A` which contains the primitive random numbers.
For the case where multiple random numbers are required, e.g. for Mixtures.
"""
function broadcast_transform_rand(m::AbstractKernelMeasure, A::NTuple{N,<:AbstractArray}) where {N}
    transform_rand.((m,), A...)
end

"""
    broadcast_transform_rand(M A)
Broadcasts the transform_rand for the measure array `M` over the array `A` which contains the primitive random numbers.
"""
function broadcast_transform_rand(M::AbstractArray{<:AbstractKernelMeasure}, A::AbstractArray)
    m = maybe_cuda(typeof(A), M)
    transform_rand.(m, A)
end

"""
    broadcast_transform_rand(M A)
Broadcasts the transform_rand for the measure array `M` over the array `A` which contains the primitive random numbers.
For the case where multiple random numbers are required, e.g. for Mixtures.
"""
function broadcast_transform_rand(M::AbstractArray{<:AbstractKernelMeasure}, A::NTuple{N,<:AbstractArray}) where {N}
    m = maybe_cuda(eltype(A), M)
    transform_rand.(m, A...)
end

"""
    rand(rng, m, dims)
Sample an Array from the measure `m` of size `dims`.
"""
function Base.rand(rng::AbstractRNG, m::AbstractKernelMeasure{T}, dims::Integer...) where {T}
    M = rand_fn(typeof(m))(rng, T, dims...)
    broadcast_transform_rand(m, M)
end

"""
    rand(rng, m, dims)
Sample an Array from the measure `m` of size `dims`.
"""
function Random.rand(rng::AbstractRNG, m::AbstractArray{<:AbstractKernelMeasure{T}}, dims::Integer...) where {T}
    M = rand_fn(eltype(m))(rng, T, dims...)
    broadcast_transform_rand(m, M)
end

"""
    rand(rng, m)
Sample an Array from the measure `m` of size 1.
"""
Base.rand(rng::AbstractRNG, d::AbstractKernelMeasure) = rand(rng, d, 1)[]

# Orthogonal methods

Base.rand(d::AbstractKernelMeasure, dims::Integer...) = rand(Random.GLOBAL_RNG, d, dims...)
Base.rand(d::AbstractKernelMeasure) = rand(Random.GLOBAL_RNG, d)

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

Base.show(io::IO, d::KernelNormal{T}) where {T} = print(io, "KernelNormal{$(T)}, μ: $(d.μ), σ: $(d.σ)")

function MeasureTheory.logdensity(d::KernelNormal{T}, x) where {T}
    μ = d.μ
    σ² = d.σ^2
    -T(0.5) * ((T(x) - μ)^2 / σ²)
end

MeasureTheory.logpdf(d::KernelNormal{T}, x) where {T<:Real} = logdensity(d, x) - log(d.σ) - log(sqrt(T(2π)))

rand_fn(::Type{<:KernelNormal}) = randn
transform_rand(d::KernelNormal, x::Real) = d.σ * x + d.μ

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

Base.show(io::IO, d::KernelExponential{T}) where {T} = print(io, "KernelExponential{$(T)}, λ: $(d.λ)")

MeasureTheory.logdensity(d::KernelExponential{T}, x) where {T} = -d.λ * T(x)
MeasureTheory.logpdf(d::KernelExponential, x) = logdensity(d, x) + log(d.λ)

rand_fn(::Type{<:KernelExponential}) = rand
transform_rand(d::KernelExponential, x::Real) = -log(x) / d.λ

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

Base.show(io::IO, d::KernelUniform{T}) where {T} = print(io, "KernelUniform{$(T)}, a: $(d.a), b: $(d.b)")

MeasureTheory.logdensity(d::KernelUniform{T}, x) where {T<:Real} = d.a <= x <= d.b ? zero(T) : -typemax(T)
MeasureTheory.logpdf(d::KernelUniform, x) = logdensity(d, x) - log(d.b - d.a)

rand_fn(::Type{<:KernelUniform}) = rand
transform_rand(d::KernelUniform, x::Real) = (d.b - d.a) * x + d.a

# KernelCircularUniform

struct KernelCircularUniform{T<:Real} <: AbstractKernelMeasure{T} end

KernelCircularUniform(::Type{T}=Float32) where {T} = KernelCircularUniform{T}()

kernel_measure(::CircularUniform, ::Type{T}=Float32) where {T} = KernelCircularUniform(T)
measure_theory(::KernelCircularUniform) = CircularUniform()
TransformVariables.as(::KernelCircularUniform) = as○

Base.show(io::IO, ::KernelCircularUniform{T}) where {T} = print(io, "KernelCircularUniform{$(T)}")

MeasureTheory.logdensity(::KernelCircularUniform{T}, x) where {T} = logdensity(KernelUniform{T}(0, 2π), x)
MeasureTheory.logpdf(d::KernelCircularUniform{T}, x) where {T} = logdensity(d, x) - log(T(2π))

rand_fn(::Type{<:KernelCircularUniform}) = rand_fn(KernelUniform)
transform_rand(::KernelCircularUniform{T}, x::Real) where {T} = transform_rand(KernelUniform{T}(0, 2π), x)

# KernelBinaryMixture

struct KernelBinaryMixture{T<:Real,U<:AbstractKernelMeasure{T},V<:AbstractKernelMeasure{T}} <: AbstractKernelMeasure{T}
    c1::U
    c2::V
    # Prefer log here, since the logdensity will be used more often than rand
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

Base.show(io::IO, d::KernelBinaryMixture{T}) where {T} = print(io, "KernelBinaryMixture{$(T)}\n  components: $(d.c1), $(d.c2) \n  log weights: $(d.log_w1), $(d.log_w2)")

MeasureTheory.logdensity(d::KernelBinaryMixture, x) = logaddexp(d.log_w1 + logpdf(d.c1, x), d.log_w2 + logpdf(d.c2, x))
MeasureTheory.logpdf(d::KernelBinaryMixture, x) = logdensity(d, x)

rand_fn(::Type{<:KernelBinaryMixture{<:Any,U,V}}) where {U,V} = (rng, T, dims...) -> (rand(rng, T, dims...), rand_fn(U)(rng, T, dims...), rand_fn(V)(rng, T, dims...))

function transform_rand(d::KernelBinaryMixture, u, x1, x2)
    if log(u) < d.log_w1
        transform_rand(d.c1, x1)
    else
        transform_rand(d.c2, x2)
    end
end

# TODO move to separate file.
# AbstractVectorizedMeasure

"""
Vectorization support by storing multiple measures in an array for broadcasting.

Implement the AbstractKernelMeasure interfaces and additionally:
- MeasureTheory.marginals():    Return the internal array of measures

You can use:
- as
- rand & rand!
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
    rand(rng, m, dims)
Sample an Array from the measure `m` of size `dims`.
"""
Base.rand(rng::AbstractRNG, d::AbstractVectorizedMeasure, dims::Integer...) = rand(rng, marginals(d), size(marginals(d))..., dims...)

"""
    rand(rng, m, dims)
Sample an Array from the measure `m` of size 1.
"""
Base.rand(rng::AbstractRNG, d::AbstractVectorizedMeasure) = rand(rng, marginals(d), size(marginals(d))..., 1)

"""
    broadcast_logdensity(d, M)
Broadcasts the logdensity function, takes care of transferring the measure to the GPU if required.
"""
function broadcast_logdensity(d::AbstractVectorizedMeasure, M)
    d = maybe_cuda(M, d)
    logdensity.(marginals(d), M)
end

"""
    broadcast_logpdf(d, M)
Broadcasts the logpdf function, takes care of transferring the measure to the GPU if required.
"""
function broadcast_logpdf(d::AbstractVectorizedMeasure, M)
    d = maybe_cuda(M, d)
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
Convert a MeasureTheory ProductMeasure into a kernel measure.
Warning: The marginals of the measure must be of the same type to transfer them to an CuArray.
"""
KernelProduct(d::ProductMeasure, ::Type{T}=Float32) where {T} = kernel_measure.(marginals(d), (T,)) |> KernelProduct

measure_theory(d::KernelProduct) = MeasureBase.productmeasure(identity, d.marginals |> Array .|> measure_theory)

MeasureTheory.marginals(d::KernelProduct) = d.marginals

Base.show(io::IO, d::KernelProduct{T}) where {T} = print(io, "KernelProduct{$(T)}\n  measures: $(typeof(d.marginals))\n  size: $(size(d.marginals))")

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
Convert a MeasureTheory ProductMeasure into a kernel measure.
Warning: The marginals of the measure must be of the same type to transfer them to an CuArray.
"""
VectorizedMeasure(d::ProductMeasure, ::Type{T}=Float32) where {T} = kernel_measure.(marginals(d), (T,)) |> VectorizedMeasure

# TODO Default behavior or prefer product measure? Behaves like a product measure if the size of the data matches the size of the marginals.
kernel_measure(d::ProductMeasure, ::Type{T}=Float32) where {T} = VectorizedMeasure(d, T)
measure_theory(d::VectorizedMeasure) = MeasureBase.productmeasure(identity, d.marginals |> Array .|> measure_theory)

MeasureTheory.marginals(d::VectorizedMeasure) = d.marginals

Base.show(io::IO, d::VectorizedMeasure{T}) where {T} = print(io, "VectorizedMeasure{$(T)}\n  internal: $(eltype(d.marginals)) \n  size: $(size(d.marginals))")

Base.size(d::VectorizedMeasure) = d.size

"""
    reduce_vectorized(op, d, M)
Reduces the first `ndims(d)` dimensions of the Matrix `M` using the operator `op`. 
"""
function reduce_vectorized(op, d::VectorizedMeasure, M::AbstractArray)
    n_red = ndims(marginals(d))
    R = reduce(op, M; dims=(1:n_red...,))
    dropdims(R; dims=(1:n_red...,))
end

function MeasureTheory.logdensity(d::VectorizedMeasure, x)
    ℓ = broadcast_logdensity(d, x)
    reduce_vectorized(+, d, ℓ)
end
function MeasureTheory.logpdf(d::VectorizedMeasure, x)
    ℓ = broadcast_logpdf(d, x)
    reduce_vectorized(+, d, ℓ)
end
