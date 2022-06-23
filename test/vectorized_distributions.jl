# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using MCMCDepth
using Bijectors
using CUDA
using Distributions
using Plots
using LinearAlgebra
using Random
using Test

const PLOT = false
maybe_histogram(x...) = PLOT ? histogram(x...) : nothing

# Yup 42 is bad style
curng = CUDA.RNG(42)
rng = Random.default_rng(42)

# WARN Different distribution types not supported only different parametrization of the same type
# ProductDistribution
gpm = ProductDistribution(fill(KernelBinaryMixture(KernelExponential(2.0), KernelNormal(10.0, 2.0), 3, 1), 50, 10))

# Correct size
@test rand(rng, gpm, 3) |> size == (50, 10, 3)
@test rand(rng, gpm) |> size == (50, 10)

# Type stability
M = @inferred rand(rng, gpm)
M = @inferred rand(curng, gpm)
gpm = to_gpu(gpm)
M = @inferred rand(curng, gpm)
M = @inferred rand(curng, gpm);
gpm = ProductDistribution(fill(KernelBinaryMixture(KernelExponential{Float64}(2.0), KernelNormal{Float64}(10.0, 2.0), 3, 1), 100, 10)) |> to_gpu
@test eltype(M) == Float64
gpm = ProductDistribution(fill(KernelBinaryMixture(KernelExponential{Float32}(2.0), KernelNormal{Float32}(10.0, 2.0), 3, 1), 100, 10)) |> to_gpu
M = @inferred rand(curng, gpm);
@test eltype(M) == Float32
gpm = ProductDistribution(fill(KernelBinaryMixture(KernelExponential{Float16}(2.0), KernelNormal{Float16}(10.0, 2.0), 3, 1), 100, 10)) |> to_gpu
M = @inferred rand(curng, gpm);
@test eltype(M) == Float16

# Use Float64 again and only vector size to compare with Distribution.jl Product
gpm = ProductDistribution(fill(KernelBinaryMixture(KernelExponential(2.0), KernelUniform(2.0, 10.0), 3, 1), 50))
product = Product([MixtureModel([Exponential(inv(2.0)), Uniform(2.0, 10.0)], normalize([3, 1], 1)) for i = 1:50])
M = rand(rng, gpm);
maybe_histogram(flatten(M))
rand(product) |> flatten |> maybe_histogram
rand(rng, gpm) |> flatten |> maybe_histogram
@inferred logdensityof(gpm, M)
@test logdensityof(gpm, M) ≈ logdensityof(product, M)
@test logdensityof(gpm, M) isa Float64

M = rand(rng, gpm, 3);
@inferred logdensityof(gpm, M)
@test logdensityof(gpm, M) isa Float64

# VectorizedDistribution
gvm = VectorizedDistribution([KernelNormal{Float64}(i, j) for i = 1:100, j = 1:10])

# Correct size
@test rand(rng, gvm, 3) |> size == (100, 10, 3)
@test rand(rng, gvm) |> size == (100, 10)

# Type stability
M = @inferred rand(rng, gvm)
M = @inferred rand(curng, gvm)
gvm = to_gpu(gvm)
M = @inferred rand(curng, gvm)
gvm = VectorizedDistribution([KernelNormal{Float64}(i, j) for i = 1:100, j = 1:10])
M = @inferred rand(curng, gvm);
@test eltype(M) == Float64
gvm = VectorizedDistribution([KernelNormal{Float32}(i, j) for i = 1:100, j = 1:10])
M = @inferred rand(curng, gvm);
@test eltype(M) == Float32
gvm = VectorizedDistribution([KernelNormal{Float16}(i, j) for i = 1:100, j = 1:10])
M = @inferred rand(curng, gvm);
@test eltype(M) == Float16

gvm = VectorizedDistribution([KernelNormal{Float64}(i, 2.0) for i = 1:100])
product = Product([Normal(i, 2.0) for i = 1:100])
M = rand(rng, gvm);
maybe_histogram(flatten(M))
rand(product) |> flatten |> maybe_histogram
rand(rng, gvm, 2) |> flatten |> maybe_histogram
@inferred logdensityof(gvm, M)
@test logdensityof(gvm, M)[] |> sum ≈ logdensityof(product, Array(M))
@test logdensityof(gvm, M) isa Array{Float64,0}
@test logdensityof(gvm, M) |> size == ()

# Test different sizes of the marginals and rand(..., dims)
gvm = @inferred VectorizedDistribution([KernelNormal{Float16}(i, 0.1) for i = 1:100])
@test ndims(gvm) == 1
M = @inferred rand(rng, gvm);
@inferred logdensityof(gvm, M)
@test logdensityof(gvm, M) |> size == ()
@test logdensityof(gvm, M) isa Array{Float16,0}
M = @inferred rand(rng, gvm, 3);
@inferred logdensityof(gvm, M)
@test logdensityof(gvm, M) |> size == (3,)
@test logdensityof(gvm, M) isa Array{Float16,1}
M = @inferred rand(rng, gvm, 3, 4);
@inferred logdensityof(gvm, M)
@test logdensityof(gvm, M) |> size == (3, 4)
@test logdensityof(gvm, M) isa Array{Float16,2}

gvm = @inferred VectorizedDistribution([KernelNormal{Float16}(i, j) for i = 1:100, j = 1:10])
@test ndims(gvm) == 2
M = @inferred rand(rng, gvm);
@inferred logdensityof(gvm, M)
@test logdensityof(gvm, M) |> size == ()
@test logdensityof(gvm, M) isa Array{Float16,0}
M = @inferred rand(rng, gvm, 3);
@inferred logdensityof(gvm, M)
@test logdensityof(gvm, M) |> size == (3,)
@test logdensityof(gvm, M) isa Array{Float16,1}
M = @inferred rand(rng, gvm, 3, 4);
@inferred logdensityof(gvm, M)
@test logdensityof(gvm, M) |> size == (3, 4)
@test logdensityof(gvm, M) isa Array{Float16,2}

# Test custom reduction dims
gvm = @inferred VectorizedDistribution([KernelNormal{Float16}(i, j) for i = 1:100, j = 1:10], 1)
@test ndims(gvm) == 1
M = @inferred rand(rng, gvm);
@inferred logdensityof(gvm, M)
@test logdensityof(gvm, M) |> size == (10,)
@test logdensityof(gvm, M) isa Array{Float16,1}
M = @inferred rand(rng, gvm, 3);
@inferred logdensityof(gvm, M)
@test logdensityof(gvm, M) |> size == (10, 3)
@test logdensityof(gvm, M) isa Array{Float16,2}
M = @inferred rand(rng, gvm, 3, 4);
@inferred logdensityof(gvm, M)
@test logdensityof(gvm, M) |> size == (10, 3, 4)
@test logdensityof(gvm, M) isa Array{Float16,3}

gvm = @inferred VectorizedDistribution([KernelNormal{Float16}(i, j) for i = 1:100, j = 1:10], 2)
@test ndims(gvm) == 1
M = @inferred rand(rng, gvm);
@inferred logdensityof(gvm, M)
@test logdensityof(gvm, M) |> size == (100,)
@test logdensityof(gvm, M) isa Array{Float16,1}
M = @inferred rand(rng, gvm, 3);
@inferred logdensityof(gvm, M)
@test logdensityof(gvm, M) |> size == (100, 3)
@test logdensityof(gvm, M) isa Array{Float16,2}
M = @inferred rand(rng, gvm, 3, 4);
@inferred logdensityof(gvm, M)
@test logdensityof(gvm, M) |> size == (100, 3, 4)
@test logdensityof(gvm, M) isa Array{Float16,3}

gvm = @inferred VectorizedDistribution([KernelNormal{Float16}(i, j) for i = 1:100, j = 1:10], (1, 2))
@test ndims(gvm) == 2
M = @inferred rand(rng, gvm);
@inferred logdensityof(gvm, M)
@test logdensityof(gvm, M) |> size == ()
@test logdensityof(gvm, M) isa Array{Float16,0}
M = @inferred rand(rng, gvm, 3);
@inferred logdensityof(gvm, M)
@test logdensityof(gvm, M) |> size == (3,)
@test logdensityof(gvm, M) isa Array{Float16,1}
M = @inferred rand(rng, gvm, 3, 4);
@inferred logdensityof(gvm, M)
@test logdensityof(gvm, M) |> size == (3, 4)
@test logdensityof(gvm, M) isa Array{Float16,2}

# Sanity check of the results, special case: dims of VectorizedDistribution == dims of data equivalent to ProductDistribution
gpm = ProductDistribution([KernelNormal{Float64}(i, j) for i = 1:100, j = 1:100]) |> to_gpu
gvm = VectorizedDistribution([KernelNormal{Float64}(i, j) for i = 1:100, j = 1:100], 1:4) |> to_gpu
M = @inferred rand(curng, gvm, 100, 5);
@test logdensityof(gpm, M) == logdensityof(gvm, M)[]

# TODO should not be required anymore if I do not use the variables
# Method ambiguities
@inferred logdensityof(KernelExponential(Float64), 100)
logdensityof_single_dist_array_data() = logdensityof.(KernelExponential(Float64), [100, 1])
@inferred logdensityof_single_dist_array_data()
logdensityof_array_dist_array_data() = logdensityof.([KernelExponential(Float64), KernelExponential(Float64)], [100, 1])
@inferred logdensityof_array_dist_array_data()
logdensityof_array_dist_single_data() = logdensityof.([KernelExponential(Float64), KernelExponential(Float64)], 100)
@inferred logdensityof_array_dist_single_data()
