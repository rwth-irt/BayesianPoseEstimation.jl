# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

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
rng = Random.default_rng(42)

# WARN Different distribution types not supported only different parametrization of the same type
# ProductDistribution

# Generate a mixture model with all parameters fixed except the mean of the normal (like the depth pixel model)
mixture_fn(μ) = KernelBinaryMixture(KernelExponential(2.0), KernelNormal(μ, 2.0), 3, 1)
dist = @inferred BroadcastedDistribution(mixture_fn, (1, 2), fill(10.0, 50, 10))
dist = @inferred BroadcastedDistribution(mixture_fn, fill(10.0, 50, 10))

# Correct size
@test rand(rng, dist, 3) |> size == (50, 10, 3)
@test rand(rng, dist) |> size == (50, 10)

# By default, Distributions.jl disallows logdensitof with multiple samples (Arrays and Matrices). BroadcastedDistribution should be inherently allowing multiple samples.
dist = @inferred BroadcastedDistribution(mixture_fn, fill(10.0, 50, 10))
X = rand(rng, dist)
@inferred logdensityof(dist, X)
X = rand(rng, dist, 3)
@inferred logdensityof(dist, X)
X = rand(rng, dist, 3, 2)
@inferred logdensityof(dist, X)
X = rand(rng, dist, 3, 2, 1)
@inferred logdensityof(dist, X)

# Correct device
dist = BroadcastedDistribution(mixture_fn, fill(10.0, 50, 10))
X = @inferred rand(rng, dist)
dist = BroadcastedDistribution(mixture_fn, CUDA.fill(10.0, 50, 10))
X = @inferred rand(curng, dist)
@test X isa CuArray{Float64,2}

# Type stability
dist = BroadcastedDistribution(mixture_fn, fill(10.0, 50, 10))
X = @inferred rand(rng, dist);
@test X isa Array{Float64,2}
dist = BroadcastedDistribution(mixture_fn, fill(10.0f0, 50, 10))
X = @inferred rand(rng, dist);
@test X isa Array{Float32,2}
dist = BroadcastedDistribution(mixture_fn, fill(Float16(10), 50, 10))
X = @inferred rand(rng, dist);
@test X isa Array{Float16,2}

# TODO CUDA.RNG also requires CUDA parameters. Will be tricky to use the correct RNG for the correct device. Probably use the correct rng based on the parameters.
dist = BroadcastedDistribution(mixture_fn, CUDA.fill(10.0, 50, 10))
X = @inferred rand(curng, dist);
@test X isa CuArray{Float64,2}
dist = BroadcastedDistribution(mixture_fn, CUDA.fill(10.0f0, 50, 10))
X = @inferred rand(curng, dist);
@test X isa CuArray{Float32,2}
dist = BroadcastedDistribution(mixture_fn, CUDA.fill(Float16(10), 50, 10))
X = @inferred rand(curng, dist);
@test X isa CuArray{Float16,2}

# Correct computation of logdensityof by comparing to Distributions.jl 
# All calculations in Float64
dist = @inferred BroadcastedDistribution(mixture_fn, fill(10.0, 500))
product = Product([MixtureModel([Exponential(inv(2.0)), Normal(10.0, 2.0)], normalize([3, 1], 1)) for i = 1:500])
rand(product) |> flatten |> maybe_histogram
rand(rng, dist) |> flatten |> maybe_histogram
X = rand(rng, dist);
@inferred logdensityof(dist, X)
@test logdensityof(dist, X) isa Array{Float64,0}
# WARN zero dimensional array
@test logdensityof(dist, X)[] ≈ logdensityof(product, X)

# Special case: number of dims equals ndims of the data → scalar value
dist = BroadcastedDistribution(mixture_fn, Dims(1:2), fill(10.0, 500))
X = rand(rng, dist, 3)
@inferred logdensityof(dist, X)
@test logdensityof(dist, X) isa Array{Float64,0}
# TODO nice, this supports multiple samples
@test logdensityof(dist, X)[] ≈ logpdf(product, X) |> sum

# VectorizedDistribution
dist = BroadcastedDistribution(mixture_fn, fill(10.0, 100, 10))

# Test different sizes of the marginals and rand(..., dims)
normal_fn(μ::T) where {T} = KernelNormal(μ, T(0.1))
dist = @inferred BroadcastedDistribution(normal_fn, [Float32(x) for x = 1:100])

dist = @inferred BroadcastedDistribution(normal_fn, Dims(1), [Float16(x) for x = 1:100])
@test ndims(dist) == 1
X = @inferred rand(rng, dist);
@inferred logdensityof(dist, X)
@test logdensityof(dist, X) |> size == ()
@test logdensityof(dist, X) isa Array{Float16,0}
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
@test logdensityof(dist, X) isa Array{Float16,0}
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
@test logdensityof(dist, M) isa Array{Float16,0}
M = @inferred rand(rng, dist, 3);
@inferred logdensityof(dist, M)
@test logdensityof(dist, M) |> size == (3,)
@test logdensityof(dist, M) isa Array{Float16,1}
M = @inferred rand(rng, dist, 3, 4);
@inferred logdensityof(dist, M)
@test logdensityof(dist, M) |> size == (3, 4)
@test logdensityof(dist, M) isa Array{Float16,2}

# TransformedDistribution - correct calculation
dist = @inferred BroadcastedDistribution(KernelExponential, [Float64(i) for i = 1:100])
t_dist = transformed(dist)
product = Product([Exponential(inv(i)) for i = 1:100])
t_product = transformed(product)

Y = @inferred rand(rng, t_dist, 3, 2, 2)
@test logpdf(t_product, Y) ≈ logdensityof(t_dist, Y)

# WARN for invlink, we need to keep the original distribution around. So use transformed(dist) inside the acceptance step
@test minimum(Y) < 0
X = @inferred invlink(dist, Y)
@test link(dist, X) ≈ Y
@test minimum(X) > 0

# By default, Distributions.jl disallows logdensitof with multiple samples (Arrays and Matrices). BroadcastedDistribution should be inherently allowing multiple samples.
dist = @inferred BroadcastedDistribution(KernelExponential, [Float64(i) for i = 1:100])
t_dist = transformed(dist)

Y = rand(rng, t_dist)
@inferred logdensityof(t_dist, Y)
Y = rand(rng, t_dist, 3)
@inferred logdensityof(t_dist, Y)
Y = rand(rng, t_dist, 3, 2)
@inferred logdensityof(t_dist, Y)
Y = rand(rng, t_dist, 3, 2, 1)
@inferred logdensityof(t_dist, Y)
