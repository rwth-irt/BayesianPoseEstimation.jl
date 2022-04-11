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

# TODO Can probably remove CUDA from it and just build on AbstractArrays and Random. Kernels should work out of the box :)
# TODO rand can specify CUDA.RNG or Random.GLOBAL_RNG for CPU / GPU execution
# TODO Rename to KernelMeasures.jl
# TODO add to_gpu(d::GpuProductMeasure) to interface? Maybe with a new supertype AbstractVectorizedMeasure


# Rename to AbstractKernelMeasure
# isbitstype types which correspond to MeasureTheory measures
"""
Implement interface:
- gpu_measure(d, ::Type{T}=Float32): Convert the MeasureTheory.jl measure to the corresponding AbstractGpuMeasure, T as parameter so we can use Float32 as default value
- cpu_measure(d): Convert the AbstractGpuMeasure to the corresponding MeasureTheory.jl measure
- `rand(rng, d::MyGpuMeasure{T})::T` create a single random number, use a CUDA.RNG for GPU execution 
- `logdensity(d::MyGpuMeasure{T}, x)::T` evaluate the unnormalized logdensity
- `logpdf(d::MyGpuMeasure{T}, x)::T` evaluate the normalized logdensity

Most of the time Float64 precision is not required, especially for GPU computations.
Thus, we default to Float32, mostly for memory capacity reasons.
"""
abstract type AbstractGpuMeasure{T} <: AbstractMeasure end

"""
Mutate `M` with random samples from the measure `d` using rng.
"""
function Random.rand!(rng::AbstractRNG, d::AbstractGpuMeasure, M::AbstractArray{T}) where {T}
    # Broadcast rand. otherwise we would get the same value in every entry of M
    # Do not put T in a tuple, CUDA compilation will fail https://github.com/JuliaGPU/CUDA.jl/issues/261 
    M .= rand.((rng,), (d,))
end

"""
Mutate `M` with random samples from the vectorized measure `d` using rng.
"""
function Random.rand!(rng::AbstractRNG, d::AbstractArray{<:AbstractGpuMeasure}, M::AbstractArray{T}) where {T}
    # Do not put T in a tuple, CUDA compilation will fail https://github.com/JuliaGPU/CUDA.jl/issues/261 
    M .= rand.((rng,), d)
end

# Non modifying rand must create array based on RNG type
ScalarOrVectorGpuMeasure{T} = Union{AbstractGpuMeasure{T},AbstractArray{<:AbstractGpuMeasure{T}}}

"""
Return an Array of type `T` with `dim` dimensions and `dims` dimensions sampled from the measure `d`.
"""
Base.rand(rng::AbstractRNG, d::AbstractGpuMeasure{T}, dim::Integer=1, dims::Integer...) where {T} = rand!(rng, d, Array{T}(undef, dim, dims...))

# WARN CUDA.RNG is not isbits?
#TODO GLOBAL_RNG seems the only one to be working, overriding RNG does not feel right
"""
Return a CUDA array of type `T` with `dim` dimensions and `dims` dimensions sampled from the measure `d`.
"""
Base.rand(::Union{CUDA.RNG,CURAND.RNG}, d::AbstractGpuMeasure{T}, dim::Integer=1, dims::Integer...) where {T} = rand!(Random.GLOBAL_RNG, d, CuArray{T}(undef, dim, dims...))

# Orthogonal rand / rand! are implemented by MeasureBase using GLOBAL_RNG 🙂

# GpuNormal

struct GpuNormal{T<:Real} <: AbstractGpuMeasure{T}
    μ::T
    σ::T
end

GpuNormal(::Type{T}=Float32) where {T} = GpuNormal{T}(0.0, 1.0)
GpuNormal(::Normal{()}, ::Type{T}=Float32) where {T} = GpuNormal(T)
GpuNormal(d::Normal{(:μ, :σ)}, ::Type{T}=Float32) where {T} = GpuNormal{T}(d.μ, d.σ)


gpu_measure(d::Normal, ::Type{T}=Float32) where {T} = GpuNormal(d, T)
cpu_measure(d::GpuNormal) = Normal(d.μ, d.σ)

Base.show(io::IO, d::GpuNormal{T}) where {T} = print(io, "GpuNormal{$(T)}, μ: $(d.μ), σ: $(d.σ)")

function MeasureTheory.logdensity(d::GpuNormal{T}, x) where {T}
    μ = d.μ
    σ² = d.σ^2
    -T(0.5) * ((T(x) - μ)^2 / σ²)
end
MeasureTheory.logpdf(d::GpuNormal{T}, x) where {T<:Real} = logdensity(d, x) - log(d.σ) - log(sqrt(T(2π)))

Base.rand(rng::AbstractRNG, d::GpuNormal{T}) where {T} = d.σ * randn(rng, T) + d.μ

TransformVariables.as(::GpuNormal) = asℝ

# GpuExponential
struct GpuExponential{T<:Real} <: AbstractGpuMeasure{T}
    λ::T
end

GpuExponential(::Type{T}=Float32) where {T} = GpuExponential{T}(1.0)
GpuExponential(::Exponential{()}, ::Type{T}=Float32) where {T} = GpuExponential(T)
GpuExponential(d::Exponential{(:λ,)}, ::Type{T}=Float32) where {T} = GpuExponential{T}(d.λ)
GpuExponential(d::Exponential{(:β,)}, ::Type{T}=Float32) where {T} = GpuExponential{T}(1 / d.β)

gpu_measure(d::Exponential, ::Type{T}=Float32) where {T} = GpuExponential(d, T)
cpu_measure(d::GpuExponential) = Exponential{(:λ,)}(d.λ)

Base.show(io::IO, d::GpuExponential{T}) where {T} = print(io, "GpuExponential{$(T)}, λ: $(d.λ)")

MeasureTheory.logdensity(d::GpuExponential{T}, x) where {T} = -d.λ * T(x)
MeasureTheory.logpdf(d::GpuExponential, x) = logdensity(d, x) + log(d.λ)

Base.rand(rng::AbstractRNG, d::GpuExponential{T}) where {T} = randexp(rng, T) / d.λ

TransformVariables.as(::GpuExponential) = asℝ₊

# GpuUniformInterval

struct GpuUniformInterval{T<:Real} <: AbstractGpuMeasure{T}
    a::T
    b::T
end

GpuUniformInterval(::Type{T}=Float32) where {T} = GpuUniformInterval{T}(0.0, 1.0)
GpuUniformInterval(::UniformInterval{()}, ::Type{T}=Float32) where {T} = GpuUniformInterval{T}(0.0, 1.0)
GpuUniformInterval(d::UniformInterval{(:a, :b)}, ::Type{T}=Float32) where {T} = GpuUniformInterval{T}(d.a, d.b)

gpu_measure(d::UniformInterval, ::Type{T}=Float32) where {T} = GpuUniformInterval(d, T)
cpu_measure(d::GpuUniformInterval) = UniformInterval(d.a, d.b)

Base.show(io::IO, d::GpuUniformInterval{T}) where {T} = print(io, "GpuUniformInterval{$(T)}, a: $(d.a), b: $(d.b)")

MeasureTheory.logdensity(d::GpuUniformInterval{T}, x) where {T<:Real} = d.a <= x <= d.b ? zero(T) : -typemax(T)
MeasureTheory.logpdf(d::GpuUniformInterval, x) = logdensity(d, x) - log(d.b - d.a)

Base.rand(rng::AbstractRNG, d::GpuUniformInterval{T}) where {T} = (d.b - d.a) * rand(rng, T) + d.a

TransformVariables.as(d::GpuUniformInterval) = as(Real, d.a, d.b)

# GpuCircularUniform

struct GpuCircularUniform{T<:Real} <: AbstractGpuMeasure{T} end

GpuCircularUniform(::Type{T}=Float32) where {T} = GpuCircularUniform{T}()

gpu_measure(::CircularUniform, ::Type{T}=Float32) where {T} = GpuCircularUniform(T)
cpu_measure(::GpuCircularUniform) = CircularUniform()

Base.show(io::IO, ::GpuCircularUniform{T}) where {T} = print(io, "GpuCircularUniform{$(T)}")

MeasureTheory.logdensity(::GpuCircularUniform{T}, x) where {T} = logdensity(GpuUniformInterval{T}(0, 2π), x)
MeasureTheory.logpdf(d::GpuCircularUniform{T}, x) where {T} = logdensity(d, x) - log(T(2π))

Base.rand(rng::AbstractRNG, ::GpuCircularUniform{T}) where {T} = rand(rng, GpuUniformInterval{T}(0, 2π))

TransformVariables.as(::GpuCircularUniform) = as○

# GpuBinaryMixture

struct GpuBinaryMixture{T<:Real,U<:AbstractGpuMeasure{T},V<:AbstractGpuMeasure{T}} <: AbstractGpuMeasure{T}
    c1::U
    c2::V
    log_w1::T
    log_w2::T
    GpuBinaryMixture(c1::U, c2::V, w1, w2) where {T,U<:AbstractGpuMeasure{T},V<:AbstractGpuMeasure{T}} = new{T,U,V}(c1, c2, Float32(log(w1 / (w1 + w2))), Float32(log(w2 / (w1 + w2))))
end

GpuBinaryMixture(d::BinaryMixture, ::Type{T}=Float32) where {T} = GpuBinaryMixture(gpu_measure(d.c1, T), gpu_measure(d.c2, T), exp(d.log_w1), exp(d.log_w2))

gpu_measure(d::BinaryMixture, ::Type{T}=Float32) where {T} = GpuBinaryMixture(d, T)
cpu_measure(d::GpuBinaryMixture) = BinaryMixture(cpu_measure(d.c1), cpu_measure(d.c2), exp(d.log_w1), exp(d.log_w2))

Base.show(io::IO, d::GpuBinaryMixture{T}) where {T} = print(io, "GpuBinaryMixture{$(T)}\n  components: $(d.c1), $(d.c2) \n  log weights: $(d.log_w1), $(d.log_w2)")

MeasureTheory.logdensity(d::GpuBinaryMixture, x) = logaddexp(d.log_w1 + logpdf(d.c1, x), d.log_w2 + logpdf(d.c2, x))
MeasureTheory.logpdf(d::GpuBinaryMixture, x) = logdensity(d, x)

function Base.rand(rng::AbstractRNG, d::GpuBinaryMixture{T}) where {T}
    log_u = log(rand(rng, T))
    if log_u < d.log_w1
        rand(rng, d.c1)
    else
        rand(rng, d.c2)
    end
end

# TODO does it make sense to introduce TransformVariables.as for Mixtures?

# AbstractVectorizedMeasure

"""
Additionally implement:
- MeasureTheory.marginals():    Return the internal array of measures

You can use:
- rand & rand!
- maybe_to_gpu(d, M)
- vectorized_logdensity(d, M)
- vectorized_logpdf(d, M)
- to_cpu(d)
- to_gpu(d)
"""
abstract type AbstractVectorizedMeasure{T} <: AbstractGpuMeasure{T} end

function maybe_to_gpu(d::AbstractVectorizedMeasure, M)
    if (M isa CuArray) && !(marginals(d) isa CuArray)
        @warn "Transferring vectorized measure to GPU, avoid overhead by calling d=to_gpu(d::GpuProductMeasure) once."
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

function vectorized_logdensity(d::AbstractVectorizedMeasure, M)
    d = maybe_to_gpu(d, M)
    logdensity.(marginals(d), M)
end

function vectorized_logpdf(d::AbstractVectorizedMeasure, M)
    d = maybe_to_gpu(d, M)
    logpdf.(marginals(d), M)
end

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

TransformVariables.as(d::AbstractVectorizedMeasure) = marginals(d) |> first |> as

# GpuProductMeasure

struct GpuProductMeasure{T<:Real,U<:AbstractGpuMeasure{T},V<:AbstractArray{U}} <: AbstractVectorizedMeasure{T}
    # WARN CUDA kernels only work for the same Measure with different parametrization
    marginals::V
end

GpuProductMeasure(d::GpuProductMeasure) = GpuProductMeasure(d.marginals)

"""
Convert a MeasureTheory ProductMeasure into a gpu measure.
Warning: The marginals of the measure must be of the same type to transfer them to an CuArray.
"""
GpuProductMeasure(d::ProductMeasure, ::Type{T}=Float32) where {T} = gpu_measure.(marginals(d), (T,)) |> GpuProductMeasure

# TODO rename to vec_measure to make clear that it does not transfer to the GPU
gpu_measure(d::ProductMeasure, ::Type{T}=Float32) where {T} = GpuProductMeasure(d, T)
cpu_measure(d::GpuProductMeasure) = MeasureBase.productmeasure(identity, d.marginals |> Array .|> cpu_measure)

MeasureTheory.marginals(d::GpuProductMeasure) = d.marginals

Base.show(io::IO, d::GpuProductMeasure{T}) where {T} = print(io, "GpuProductMeasure{$(T)}\n  measures: $(typeof(d.marginals))\n  size: $(size(d.marginals))")

function MeasureTheory.logdensity(d::GpuProductMeasure, x)
    ℓ = vectorized_logdensity(d, x)
    sum(ℓ)
end
function MeasureTheory.logpdf(d::GpuProductMeasure, x)
    ℓ = vectorized_logpdf(d, x)
    sum(ℓ)
end

# GpuVectorizedMeasure
# TODO remove VectorizedMeasure since I can abstract away the CUDA part
"""
    GpuVectorizedMeasure
Behaves similar to the ProductMeasure but assumes a vectorization of the data over last dimension.
"""
struct GpuVectorizedMeasure{T<:Real,U<:AbstractGpuMeasure{T},V<:AbstractArray{U}} <: AbstractVectorizedMeasure{T}
    marginals::V
end

"""
Convert a MeasureTheory ProductMeasure into a gpu measure.
Warning: The marginals of the measure must be of the same type to transfer them to an CuArray.
"""
GpuVectorizedMeasure(d::ProductMeasure, ::Type{T}=Float32) where {T} = gpu_measure(d, T) |> marginals |> GpuVectorizedMeasure

MeasureTheory.marginals(d::GpuVectorizedMeasure) = d.marginals

Base.show(io::IO, d::GpuVectorizedMeasure{T}) where {T} = print(io, "GpuVectorizedMeasure{$(T)}\n  internal: $(eltype(d.marginals)) \n  size: $(size(d.marginals))")

Base.size(d::GpuVectorizedMeasure) = d.size

"""
    reduce_to_last_dim(op, M)
Reduces all dimensions but the last one.
"""
function reduce_to_last_dim(op, M::AbstractArray{<:Any,N}) where {N}
    R = reduce(op, M; dims=(1:N-1...,))
    dropdims(R; dims=(1:N-1...,))
end

function MeasureTheory.logdensity(d::GpuVectorizedMeasure, x)
    ℓ = vectorized_logdensity(d, x)
    reduce_to_last_dim(+, ℓ)
end
function MeasureTheory.logpdf(d::GpuVectorizedMeasure, x)
    ℓ = vectorized_logpdf(d, x)
    reduce_to_last_dim(+, ℓ)
end
