# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using Bijectors
using CUDA
using Distributions
using LinearAlgebra
using MCMCDepth
using Plots
using Random
using Test

const PLOT = true
maybe_histogram(x...) = PLOT ? histogram(x...) : nothing

# Yup 42 is bad style
curng = CUDA.RNG(42)
CUDA.allowscalar(false)
rng = Random.default_rng(42)

# WARN Different distribution types not supported only different parametrization of the same type
# ProductDistribution

# Generate a mixture model with all parameters fixed except the mean of the normal (like the depth pixel model)
mixture_fn(μ) = KernelBinaryMixture(KernelExponential(2.0), KernelNormal(μ, 2.0), 3.0, 1.0)
bm = @inferred mixture_fn(1.0)
@test exp(bm.log_weight_1) == 3.0 / 4
@test exp(bm.log_weight_2) == 1.0 / 4
dist = @inferred BroadcastedDistribution(mixture_fn, (1,), fill(10.0, 50, 10))
dist = @inferred ProductBroadcastedDistribution(mixture_fn, fill(10.0, 50, 10))

# Correct size
@test rand(rng, dist, 3) |> size == (50, 10, 3)
@test rand(rng, dist) |> size == (50, 10)

# By default, Distributions.jl disallows logdensityof with multiple samples (Arrays and Matrices). BroadcastedDistribution should be inherently allowing multiple samples.
dist = @inferred ProductBroadcastedDistribution(mixture_fn, fill(10.0, 50, 10))
X = rand(rng, dist)
@inferred logdensityof(dist, X)
X = rand(rng, dist, 3)
@inferred logdensityof(dist, X)
X = rand(rng, dist, 3, 2)
@inferred logdensityof(dist, X)
X = rand(rng, dist, 3, 2, 1)
@inferred logdensityof(dist, X)

# Correct device
dist = @inferred ProductBroadcastedDistribution(mixture_fn, fill(10.0, 50, 10))
X = @inferred rand(rng, dist, 2)
@test X isa Array{Float64,3}
ℓ = @inferred logdensityof(dist, X)
@test ℓ isa Array{Float64,1}

dist = @inferred ProductBroadcastedDistribution(mixture_fn, CUDA.fill(10.0, 50, 10))
X = @inferred rand(curng, dist, 2)
@test X isa CuArray{Float64,3}
ℓ = @inferred logdensityof(dist, X)
@test ℓ isa CuArray{Float64,1}

# Type stability
dist = @inferred ProductBroadcastedDistribution(mixture_fn, fill(10.0, 50, 10))
X = @inferred rand(rng, dist);
@test X isa Array{Float64,2}
dist = @inferred ProductBroadcastedDistribution(mixture_fn, fill(10.0f0, 50, 10))
X = @inferred rand(rng, dist);
@test X isa Array{Float32,2}
dist = @inferred ProductBroadcastedDistribution(mixture_fn, fill(Float16(10), 50, 10))
X = @inferred rand(rng, dist);
@test X isa Array{Float16,2}

# TODO CUDA.RNG also requires CUDA parameters. Will be tricky to use the correct RNG for the correct device. Probably use the correct rng based on the parameters.
dist = @inferred ProductBroadcastedDistribution(mixture_fn, CUDA.fill(10.0, 50, 10))
X = @inferred rand(curng, dist);
@test X isa CuArray{Float64,2}
dist = @inferred ProductBroadcastedDistribution(mixture_fn, CUDA.fill(10.0f0, 50, 10))
X = @inferred rand(curng, dist);
@test X isa CuArray{Float32,2}
dist = @inferred ProductBroadcastedDistribution(mixture_fn, CUDA.fill(Float16(10), 50, 10))
X = @inferred rand(curng, dist);
@test X isa CuArray{Float16,2}

# Correct computation of logdensityof by comparing to Distributions.jl 
# All calculations in Float64
dist = @inferred ProductBroadcastedDistribution(mixture_fn, fill(10.0, 500))
product = Product([MixtureModel([Exponential(2.0), Normal(10.0, 2.0)], normalize([3, 1], 1)) for i = 1:500])
rand(product) |> flatten |> maybe_histogram
rand(rng, dist) |> flatten |> maybe_histogram
X = rand(rng, dist);
@inferred logdensityof(dist, X)
@test logdensityof(dist, X) isa Float64
# WARN zero dimensional array
@test logdensityof(dist, X) ≈ logdensityof(product, X)

# Special case: number of dims equals ndims of the data → scalar value
dist = @inferred BroadcastedDistribution(mixture_fn, Dims(1:2), fill(10.0, 500))
X = rand(rng, dist, 3)
@inferred logdensityof(dist, X)
@test logdensityof(dist, X) isa Float64
@test logdensityof(dist, X) ≈ logpdf(product, X) |> sum

# Should also work with scalars
dist = @inferred ProductBroadcastedDistribution(KernelExponential, Float16(10.0))
x = @inferred rand(rng, dist)
@test x isa Float16
ℓ = @inferred logdensityof(dist, x)
@test ℓ isa Float16
X = rand(curng, dist, 3)
@test X isa CuArray{Float16,1}
@test size(X) == (3,)
ℓ = @inferred logdensityof(dist, X)
@test ℓ isa CuArray{Float16,1}

# Test different sizes of the marginals and rand(..., dims)
normal_fn(μ::T) where {T} = KernelNormal(μ, T(0.1))
dist = @inferred ProductBroadcastedDistribution(normal_fn, [Float32(x) for x = 1:100])

dist = @inferred BroadcastedDistribution(normal_fn, Dims(1), [Float16(x) for x = 1:100])
@test ndims(dist) == 1
X = @inferred rand(rng, dist);
@inferred logdensityof(dist, X)
@test logdensityof(dist, X) |> size == ()
@test logdensityof(dist, X) isa Float16
X = @inferred rand(rng, dist, 3);
@inferred logdensityof(dist, X)
@test logdensityof(dist, X) |> size == (3,)
@test logdensityof(dist, X) isa Array{Float16,1}
X = @inferred rand(rng, dist, 3, 4);
@inferred logdensityof(dist, X)
@test logdensityof(dist, X) |> size == (3, 4)
@test logdensityof(dist, X) isa Array{Float16,2}

dist = @inferred BroadcastedDistribution(KernelNormal, (1, 2), [Float16(i) for i = 1:100], [Float16(j) for i = 1:100, j = 1:10])
@test ndims(dist) == 2
X = @inferred rand(rng, dist);
@inferred logdensityof(dist, X)
@test logdensityof(dist, X) |> size == ()
@test logdensityof(dist, X) isa Float16
X = @inferred rand(rng, dist, 3);
@inferred logdensityof(dist, X)
@test logdensityof(dist, X) |> size == (3,)
@test logdensityof(dist, X) isa Array{Float16,1}
X = @inferred rand(rng, dist, 3, 4);
@inferred logdensityof(dist, X)
@test logdensityof(dist, X) |> size == (3, 4)
@test logdensityof(dist, X) isa Array{Float16,2}

# Test custom reduction dims
dist = @inferred BroadcastedDistribution(KernelNormal, (1,), [Float16(i) for i = 1:100], [Float16(j) for i = 1:100, j = 1:10])
@test ndims(dist) == 1
M = @inferred rand(rng, dist);
@inferred logdensityof(dist, M)
@test logdensityof(dist, M) |> size == (10,)
@test logdensityof(dist, M) isa Array{Float16,1}
M = @inferred rand(rng, dist, 3);
@inferred logdensityof(dist, M)
@test logdensityof(dist, M) |> size == (10, 3)
@test logdensityof(dist, M) isa Array{Float16,2}
M = @inferred rand(rng, dist, 3, 4);
@inferred logdensityof(dist, M)
@test logdensityof(dist, M) |> size == (10, 3, 4)
@test logdensityof(dist, M) isa Array{Float16,3}

dist = @inferred BroadcastedDistribution(KernelNormal, (2,), [Float16(i) for i = 1:100], [Float16(j) for i = 1:100, j = 1:10])
@test ndims(dist) == 1
M = @inferred rand(rng, dist);
@inferred logdensityof(dist, M)
@test logdensityof(dist, M) |> size == (100,)
@test logdensityof(dist, M) isa Array{Float16,1}
M = @inferred rand(rng, dist, 3);
@inferred logdensityof(dist, M)
@test logdensityof(dist, M) |> size == (100, 3)
@test logdensityof(dist, M) isa Array{Float16,2}
M = @inferred rand(rng, dist, 3, 4);
@inferred logdensityof(dist, M)
@test logdensityof(dist, M) |> size == (100, 3, 4)
@test logdensityof(dist, M) isa Array{Float16,3}

dist = @inferred BroadcastedDistribution(KernelNormal, (1, 2), [Float16(i) for i = 1:100], [Float16(j) for i = 1:100, j = 1:10])
@test ndims(dist) == 2
M = @inferred rand(rng, dist);
@inferred logdensityof(dist, M)
@test logdensityof(dist, M) |> size == ()
@test logdensityof(dist, M) isa Float16
M = @inferred rand(rng, dist, 3);
@inferred logdensityof(dist, M)
@test logdensityof(dist, M) |> size == (3,)
@test logdensityof(dist, M) isa Array{Float16,1}
M = @inferred rand(rng, dist, 3, 4);
@inferred logdensityof(dist, M)
@test logdensityof(dist, M) |> size == (3, 4)
@test logdensityof(dist, M) isa Array{Float16,2}

# TransformedDistribution
dist = @inferred BroadcastedDistribution(KernelExponential, (1,), [Float64(i) for i = 1:100])
t_dist = transformed(dist)
product = Product([Exponential(i) for i = 1:100])
t_product = transformed(product)

@test Broadcast.materialize(bijector(dist)).bijectors |> eltype <: Bijectors.Log

# correct calculation
Y = @inferred rand(rng, t_dist, 3, 2, 2)
@test logpdf(t_product, Y) ≈ logdensityof(t_dist, Y)

# WARN for invlink, we need to keep the original distribution around. So use transformed(dist) inside the acceptance step
@test minimum(Y) < 0
X = @inferred invlink(dist, Y)
@test link(dist, X) ≈ Y
@test minimum(X) > 0

# By default, Distributions.jl disallows logdensitof with multiple samples (Arrays and Matrices). BroadcastedDistribution should be inherently allowing multiple samples.
dist = @inferred ProductBroadcastedDistribution(KernelExponential, [Float64(i) for i = 1:100])
t_dist = transformed(dist)

Y = rand(rng, t_dist)
@inferred logdensityof(t_dist, Y)
Y = rand(rng, t_dist, 3)
@inferred logdensityof(t_dist, Y)
Y = rand(rng, t_dist, 3, 2)
@inferred logdensityof(t_dist, Y)
Y = rand(rng, t_dist, 3, 2, 1)
@inferred logdensityof(t_dist, Y)

X_invlink = @inferred invlink(dist, Y)
b = @inferred inverse(bijector(dist))
X, logjac = @inferred with_logabsdet_jacobian(b, Y)
@test X_invlink == X
@test logjac == logabsdetjac(b, Y)
@test logdensityof(t_dist, Y) ≈ logdensityof(dist, X) + logjac

# CUDA
cudist = @inferred ProductBroadcastedDistribution(KernelExponential, CuArray([Float64(i) for i = 1:100]))
t_cudist = transformed(cudist)

@test Broadcast.materialize(bijector(cudist)).bijectors |> eltype <: Bijectors.Log

# correct calculation
Y = @inferred rand(curng, t_cudist, 3, 2, 2)
@test logpdf(t_product, Array(Y)) ≈ logdensityof(t_cudist, Y) |> Array

# Bijector on CUDA
X_invlink = @inferred invlink(cudist, Y)
b = @inferred inverse(bijector(cudist))
X, logjac = @inferred with_logabsdet_jacobian(b, Y)
@test X_invlink == X
@test logjac == logabsdetjac(b, Y)
@test logdensityof(t_cudist, Y) ≈ logdensityof(cudist, X) + logjac

# WARN for invlink, we need to keep the original distribution around. So use transformed(dist) inside the acceptance step
@test minimum(Y) < 0
X = @inferred invlink(cudist, Y)
@test link(cudist, X) ≈ Y
@test minimum(X) > 0

# Test if the constructor is executed on the GPU, evaluating the support of the marginals is tricky
B = @inferred ProductBroadcastedDistribution(KernelExponential, CUDA.fill(10.0, 1000))
B = @inferred BroadcastedDistribution(KernelExponential, (1,), CUDA.fill(10.0, 1000))
X = rand(CUDA.default_rng(), B, 2)
logdensityof(B, X)
