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
using LinearAlgebra
using Plots
using Random
using Test

const PLOT = false
maybe_histogram(x...) = PLOT ? histogram(x...) : nothing

# Yup 42 is bad style
curng = CUDA.RNG(42)
rng = Random.default_rng(42)

# Correct size
@test rand(rng, KernelNormal(), 100, 100) |> size == (100, 100)
@test rand(rng, KernelNormal()) |> size == ()
@test rand(rng, KernelNormal(Float16)) isa Float16
# Size for arrays of distributions
@test rand(rng, fill(KernelNormal(), 100), 100, 100) |> size == (100, 100, 100)
@test rand(rng, fill(KernelNormal(), 100, 100), 100) |> size == (100, 100, 100)
@test rand(rng, fill(KernelNormal(), 100, 100)) |> size == (100, 100)

# Correct device for RNG
@test rand(CUDA.RNG(), KernelNormal(), 100, 100) isa CuArray
@test rand(CUDA.RNG(), fill(KernelNormal(), 100), 100, 100) isa CuArray
@test rand(CURAND.RNG(), KernelNormal(), 100, 100) isa CuArray
@test rand(CURAND.RNG(), fill(KernelNormal(), 100), 100, 100) isa CuArray
@test rand(Random.default_rng(), KernelNormal(), 100, 100) isa Array
@test rand(Random.default_rng(), fill(KernelNormal(), 100), 100, 100) isa Array
@test rand(MersenneTwister(), KernelNormal(), 100, 100) isa Array

@test rand!(CUDA.RNG(), KernelNormal(), Array{Float16}(undef, 100, 100)) isa Array{Float16}
@test rand!(CURAND.RNG(), KernelNormal(), Array{Float16}(undef, 100, 100)) isa Array{Float16}
@test rand!(Random.default_rng(), KernelNormal(), Array{Float16}(undef, 100, 100)) isa Array{Float16}
@test rand!(MersenneTwister(), KernelNormal(), Array{Float16}(undef, 100, 100)) isa Array{Float16}

@test rand!(CUDA.RNG(), KernelNormal(), CuArray{Float16}(undef, 100, 100)) isa CuArray{Float16}
@test rand!(CURAND.RNG(), KernelNormal(), CuArray{Float16}(undef, 100, 100)) isa CuArray{Float16}
@test rand!(Random.default_rng(), KernelNormal(), CuArray{Float16}(undef, 100, 100)) isa CuArray{Float16}
@test rand!(MersenneTwister(), KernelNormal(), CuArray{Float16}(undef, 100, 100)) isa CuArray{Float16}

# KernelNormal
M = @inferred rand(curng, KernelNormal(Float64), 100, 100)
@test eltype(M) == Float64
M = @inferred rand(curng, KernelNormal(Float32), 100, 100)
@test eltype(M) == Float32
M = @inferred rand(curng, KernelNormal(), 100, 100)
@test eltype(M) == Float32
M = @inferred rand(curng, KernelNormal(Float16), 100, 100)
@test eltype(M) == Float16

normal = Normal(10.0, 2.0)
gn = KernelNormal(10.0, 2.0)

@test maximum(KernelNormal(Float16)) == Inf16
@test minimum(KernelNormal(Float16)) == -Inf16
@test insupport(KernelNormal(Float16), 0)
@test insupport(KernelNormal(Float16), Inf)
@test insupport(KernelNormal(Float16), -Inf)
@test bijector(KernelNormal()) == bijector(Normal())

M = rand(rng, gn, 100, 100)
maybe_histogram(flatten(M))
maybe_histogram([rand(normal) for _ in 1:100*100])
logdensity_gn_M(gn, M) = logdensityof.(gn, M)
@inferred logdensity_gn_M(gn, M)
@test logdensityof(gn, 1.0) == logdensityof(normal, 1.0)

# KernelExponential
M = @inferred rand(curng, KernelExponential(Float64), 100, 100)
@test eltype(M) == Float64
M = @inferred rand(curng, KernelExponential(Float32), 100, 100)
@test eltype(M) == Float32
M = @inferred rand(curng, KernelExponential(), 100, 100)
@test eltype(M) == Float32
M = @inferred rand(curng, KernelExponential(Float16), 100, 100)
@test eltype(M) == Float16

exponential = Exponential(10.0)
ge = KernelExponential(10.0)

@test maximum(KernelExponential(Float16)) == Inf16
@test minimum(KernelExponential(Float16)) == 0
@test insupport(KernelExponential(Float16), 0)
@test insupport(KernelExponential(Float16), Inf)
@test !insupport(KernelExponential(Float16), -eps(Float16))
@test bijector(KernelExponential()) == bijector(Exponential())

M = rand(rng, ge, 100, 100)
maybe_histogram(flatten(M))
maybe_histogram([rand(exponential) for _ in 1:100*100])
logdensity_ge_M(ge, M) = logdensityof.(ge, M)
@inferred logdensity_ge_M(ge, M)
@test logdensityof(ge, 1.0) == logdensityof(exponential, 1.0)
@test logdensityof(ge, 0.0) == logdensityof(exponential, 0.0)
@test logdensityof(ge, -1.0) == logdensityof(exponential, -1.0)

# KernelUniform
M = @inferred rand(curng, KernelUniform(Float64), 100, 100)
@test eltype(M) == Float64
M = @inferred rand(curng, KernelUniform(Float32), 100, 100)
@test eltype(M) == Float32
M = @inferred rand(curng, KernelUniform(), 100, 100)
@test eltype(M) == Float32
M = @inferred rand(curng, KernelUniform(Float16), 100, 100)
@test eltype(M) == Float16

gu = KernelUniform(5.0, 10.0)
uniform = Uniform(5.0, 10.0)

@test maximum(KernelUniform{Float16}(1, 10)) == Float16(10)
@test minimum(KernelUniform{Float16}(1, 10)) == Float16(1)
@test insupport(KernelUniform{Float16}(1, 10), 1)
@test insupport(KernelUniform{Float16}(1, 10), 10)
@test !insupport(KernelUniform{Float32}(1, 10), 10.001)
@test !insupport(KernelUniform{Float32}(1, 10), 0.999)
@test bijector(KernelUniform(Int64)) == bijector(Uniform())
@test bijector(KernelUniform(Float64)) == bijector(Uniform())
@test bijector(KernelUniform(1.0, 10.0)) == bijector(Uniform(1.0, 10.0))

M = rand(rng, gu, 100, 100)
maybe_histogram(flatten(M))
maybe_histogram([rand(uniform) for _ in 1:100*100])
logdensity_gu_M(gu, M) = logdensityof.(gu, M)
@inferred logdensity_gu_M(gu, M)
@test logdensityof(gu, 0.5) == logdensityof(uniform, 0.5)
@test logdensityof(gu, 1.5) == logdensityof(uniform, 1.5)

# KernelCircularUniform
M = @inferred rand(curng, KernelCircularUniform(Float64), 100, 100)
@test eltype(M) == Float64
M = @inferred rand(curng, KernelCircularUniform(Float32), 100, 100)
@test eltype(M) == Float32
M = @inferred rand(curng, KernelCircularUniform(), 100, 100)
@test eltype(M) == Float32
M = @inferred rand(curng, KernelCircularUniform(Float16), 100, 100)
@test eltype(M) == Float16

gcu = KernelCircularUniform(Float64)
pseudo_circular = Uniform(0, 2π)

@test maximum(KernelCircularUniform{Float16}()) == Float16(2π)
@test minimum(KernelCircularUniform{Float16}()) == Float16(0)
@test insupport(KernelCircularUniform{Float16}(), Float16(2π))
@test insupport(KernelCircularUniform{Float16}(), 0)
@test !insupport(KernelCircularUniform{Float32}(), -0001)
@test !insupport(KernelCircularUniform{Float32}(), 2π + 0.001)
@test bijector(KernelCircularUniform(Int64)) == Circular{0}()

M = rand(rng, gcu, 100, 100)
maybe_histogram(flatten(M))
maybe_histogram([rand(pseudo_circular) for _ in 1:100*100])
logdensity_gcu_M(gcu, M) = logdensityof.(gcu, M)
@inferred logdensity_gcu_M(gcu, M)
@test logdensityof(gcu, 0.5) == logdensityof(pseudo_circular, 0.5)
@test logdensityof(gcu, 1.5) == logdensityof(pseudo_circular, 1.5)

# KernelBinaryMixture
gbm = KernelBinaryMixture(KernelExponential{Float64}(2.0), KernelUniform{Float64}(2.0, 10.0), 3, 1)
bm = MixtureModel([Exponential(2.0), Uniform(2.0, 10.0)], normalize([3, 1], 1))

M = @inferred rand(curng, KernelBinaryMixture(KernelExponential(2.0), KernelNormal{Float64}(10.0, 2), 3, 1), 100, 100)
@test eltype(M) == Float64
M = @inferred rand(curng, KernelBinaryMixture(KernelExponential{Float32}(2.0), KernelNormal{Float32}(10.0, 2), 3, 1), 100, 100)
@test eltype(M) == Float32
M = @inferred rand(curng, KernelBinaryMixture(KernelExponential{Float16}(2.0), KernelNormal{Float16}(10.0, 2), 3, 1), 100, 100)
@test eltype(M) == Float16

@test maximum(KernelBinaryMixture(KernelExponential{Float64}(2.0), KernelUniform{Float64}(1.0, 2.0), 3, 1)) == Inf
@test maximum(KernelBinaryMixture(KernelUniform{Float64}(2.0, 3.0), KernelUniform{Float64}(1.0, 2.0), 3, 1)) == 3
@test minimum(KernelBinaryMixture(KernelUniform{Float64}(2.0, 3.0), KernelUniform{Float64}(1.0, 2.0), 3, 1)) == 1
@test minimum(KernelBinaryMixture(KernelUniform{Float64}(2.0, 3.0), KernelNormal{Float64}(1.0, 2.0), 3, 1)) == -Inf
@test insupport(KernelBinaryMixture(KernelUniform{Float64}(2.0, 3.0), KernelUniform{Float64}(1.0, 2.0), 3, 1), 1.0)
@test insupport(KernelBinaryMixture(KernelUniform{Float64}(2.0, 3.0), KernelUniform{Float64}(1.0, 2.0), 3, 1), 3.0)
@test !insupport(KernelBinaryMixture(KernelUniform{Float64}(2.0, 3.0), KernelUniform{Float64}(1.0, 2.0), 3, 1), 0.99999)
@test !insupport(KernelBinaryMixture(KernelUniform{Float64}(2.0, 3.0), KernelUniform{Float64}(1.0, 2.0), 3, 1), 3.0001)
@test bijector(KernelBinaryMixture(KernelUniform{Float64}(2.0, 3.0), KernelUniform{Float64}(1.0, 2.0), 3, 1)) == Bijectors.TruncatedBijector(1.0, 3.0)
@test bijector(KernelBinaryMixture(KernelExponential{Float64}(2.0), KernelUniform{Float64}(-1.0, 2.0), 3, 1)) == Bijectors.TruncatedBijector(-1.0, Inf)
@test bijector(KernelBinaryMixture(KernelExponential{Float64}(2.0), KernelUniform{Float64}(1.0, 2.0), 3, 1)) == Bijectors.TruncatedBijector(0.0, Inf)
@test bijector(KernelBinaryMixture(KernelUniform{Float64}(-Inf, 1.0), KernelUniform{Float64}(1.0, 2.0), 3, 1)) == Bijectors.TruncatedBijector(-Inf, 2.0)
@test bijector(KernelBinaryMixture(KernelUniform{Float64}(-Inf, 1.0), KernelUniform{Float64}(1.0, Inf), 3, 1)) == Bijectors.TruncatedBijector(-Inf, Inf)

M = rand(rng, gbm, 100, 100);
maybe_histogram(flatten(M))
maybe_histogram([rand(bm) for _ in 1:100*100])
logdensity_gbm_M(gbm, M) = logdensityof.(gbm, M)
@inferred logdensity_gbm_M(gbm, M)
@inferred logdensityof(gbm, 1.0)
@test logdensityof(gbm, 1.0) ≈ logdensityof(bm, 1.0)
@test logdensityof(gbm, 10.0) ≈ logdensityof(bm, 10.0)
@test logdensityof(gbm, 100.0) ≈ logdensityof(bm, 100.0)
@test logdensityof(gbm, -1.0) ≈ logdensityof(bm, -1.0)
# TODO numerical unstable for small numbers?
# @test logdensityof(gbm, 0.01) ≈ logdensityof(bm, 0.01)
