# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using MCMCDepth
using Bijectors
using CUDA, MeasureTheory
include("MeasureTheoryExtensions.jl")
using Plots
using Random
using Test

const PLOT = false
maybe_histogram(x...) = PLOT ? histogram(x...) : nothing

# Yup 42 is bad style, like this emoji 🤣
curng = CUDA.RNG(42)
rng = Random.default_rng(42)

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

normal = MeasureTheory.Normal(10.0, 2.0)
gn = KernelNormal(10.0, 2.0)

@test maximum(KernelNormal(Float16)) == Inf16
@test minimum(KernelNormal(Float16)) == -Inf16
@test MCMCDepth.insupport(KernelNormal(Float16), 0)
@test MCMCDepth.insupport(KernelNormal(Float16), Inf)
@test MCMCDepth.insupport(KernelNormal(Float16), -Inf)
@test bijector(KernelNormal()) == bijector(Dists.Normal())

M = rand(rng, gn, 100, 100)
maybe_histogram(flatten(M))
maybe_histogram([rand(normal) for _ in 1:100*100])
@inferred logdensityof(gn, M)
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

# WARN MeasureTheory uses β=1/λ by default
exponential = MeasureTheory.Exponential(λ=0.1)
ge = KernelExponential(0.1)

@test maximum(KernelExponential(Float16)) == Inf16
@test minimum(KernelExponential(Float16)) == 0
@test MCMCDepth.insupport(KernelExponential(Float16), 0)
@test MCMCDepth.insupport(KernelExponential(Float16), Inf)
@test !MCMCDepth.insupport(KernelExponential(Float16), -eps(Float16))
@test bijector(KernelExponential()) == bijector(Dists.Exponential())

M = rand(rng, ge, 100, 100)
maybe_histogram(flatten(M))
maybe_histogram([rand(exponential) for _ in 1:100*100])
@inferred logdensityof(ge, M)
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
uniform = UniformInterval(5.0, 10.0)

@test maximum(KernelUniform{Float16}(1, 10)) == Float16(10)
@test minimum(KernelUniform{Float16}(1, 10)) == Float16(1)
@test MCMCDepth.insupport(KernelUniform{Float16}(1, 10), 1)
@test MCMCDepth.insupport(KernelUniform{Float16}(1, 10), 10)
@test !MCMCDepth.insupport(KernelUniform{Float32}(1, 10), 10.001)
@test !MCMCDepth.insupport(KernelUniform{Float32}(1, 10), 0.999)
@test bijector(KernelUniform(Int64)) == bijector(Dists.Uniform())
@test bijector(KernelUniform(Float64)) == bijector(Dists.Uniform())
@test bijector(KernelUniform(1.0, 10.0)) == bijector(Dists.Uniform(1.0, 10.0))

M = rand(rng, gu, 100, 100)
maybe_histogram(flatten(M))
maybe_histogram([rand(uniform) for _ in 1:100*100])
@inferred logdensityof(gu, M)
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
circular_uniform = CircularUniform()

@test maximum(KernelCircularUniform{Float16}()) == Float16(2π)
@test minimum(KernelCircularUniform{Float16}()) == Float16(0)
@test MCMCDepth.insupport(KernelCircularUniform{Float16}(), Float16(2π))
@test MCMCDepth.insupport(KernelCircularUniform{Float16}(), 0)
@test !MCMCDepth.insupport(KernelCircularUniform{Float32}(), -0001)
@test !MCMCDepth.insupport(KernelCircularUniform{Float32}(), 2π + 0.001)
@test bijector(KernelCircularUniform(Int64)) == Circular{0}()

M = rand(rng, gcu, 100, 100)
maybe_histogram(flatten(M))
maybe_histogram([rand(circular_uniform) for _ in 1:100*100])
@inferred logdensityof(gcu, M)
@test logdensityof(gcu, 0.5) == logdensityof(circular_uniform, 0.5)
@test logdensityof(gcu, 1.5) == logdensityof(circular_uniform, 1.5)

# KernelBinaryMixture
gbm = KernelBinaryMixture(KernelExponential{Float64}(2.0), KernelUniform{Float64}(2.0, 10.0), 3, 1)
bm = BinaryMixture(MeasureTheory.Exponential(λ=2.0), UniformInterval(2.0, 10.0), 3, 1)

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
@test MCMCDepth.insupport(KernelBinaryMixture(KernelUniform{Float64}(2.0, 3.0), KernelUniform{Float64}(1.0, 2.0), 3, 1), 1.0)
@test MCMCDepth.insupport(KernelBinaryMixture(KernelUniform{Float64}(2.0, 3.0), KernelUniform{Float64}(1.0, 2.0), 3, 1), 3.0)
@test !MCMCDepth.insupport(KernelBinaryMixture(KernelUniform{Float64}(2.0, 3.0), KernelUniform{Float64}(1.0, 2.0), 3, 1), 0.99999)
@test !MCMCDepth.insupport(KernelBinaryMixture(KernelUniform{Float64}(2.0, 3.0), KernelUniform{Float64}(1.0, 2.0), 3, 1), 3.0001)
@test bijector(KernelBinaryMixture(KernelUniform{Float64}(2.0, 3.0), KernelUniform{Float64}(1.0, 2.0), 3, 1)) == Bijectors.TruncatedBijector(1.0, 3.0)
@test bijector(KernelBinaryMixture(KernelExponential{Float64}(2.0), KernelUniform{Float64}(-1.0, 2.0), 3, 1)) == Bijectors.TruncatedBijector(-1.0, Inf)
@test bijector(KernelBinaryMixture(KernelExponential{Float64}(2.0), KernelUniform{Float64}(1.0, 2.0), 3, 1)) == Bijectors.TruncatedBijector(0.0, Inf)
@test bijector(KernelBinaryMixture(KernelUniform{Float64}(-Inf, 1.0), KernelUniform{Float64}(1.0, 2.0), 3, 1)) == Bijectors.TruncatedBijector(-Inf, 2.0)
@test bijector(KernelBinaryMixture(KernelUniform{Float64}(-Inf, 1.0), KernelUniform{Float64}(1.0, Inf), 3, 1)) == Bijectors.TruncatedBijector(-Inf, Inf)


M = rand(rng, gbm, 100, 100);
maybe_histogram(flatten(M))
maybe_histogram([rand(bm) for _ in 1:100*100])
@inferred logdensityof(gbm, M)
@inferred logdensityof(gbm, 1.0)
@test logdensityof(gbm, 1.0) ≈ logdensityof(bm, 1.0)
@test logdensityof(gbm, 10.0) ≈ logdensityof(bm, 10.0)
@test logdensityof(gbm, 100.0) ≈ logdensityof(bm, 100.0)
@test logdensityof(gbm, -1.0) ≈ logdensityof(bm, -1.0)
# TODO numerical unstable for small numbers?
# @test logdensityof(gbm, 0.01) ≈ logdensityof(bm, 0.01)

# TODO move
# WARN Different measure types not supported only different parametrization of the same type
# ProductDistribution
pm = For(100, 100) do i, j
    BinaryMixture(MeasureTheory.Exponential(λ=2.0), MeasureTheory.Normal(10.0, 2), 3, 1)
end;
gpm = ProductDistribution(fill(KernelBinaryMixture(KernelExponential(2.0), KernelNormal(10.0, 2.0), 3, 1), 100, 100))
M = @inferred rand(rng, gpm)
M = @inferred rand(curng, gpm)
gpm = to_gpu(gpm)
M = @inferred rand(curng, gpm)
gpm = ProductDistribution(fill(KernelBinaryMixture(KernelExponential(2.0), KernelNormal(10.0, 2.0), 3, 1), 100, 100)) |> to_gpu
M = @inferred rand(curng, gpm);
@test eltype(M) == Float64
gpm = ProductDistribution(fill(KernelBinaryMixture(KernelExponential{Float32}(2.0), KernelNormal{Float32}(10.0, 2.0), 3, 1), 100, 100)) |> to_gpu
M = @inferred rand(curng, gpm);
@test eltype(M) == Float32
gpm = ProductDistribution(fill(KernelBinaryMixture(KernelExponential{Float16}(2.0), KernelNormal{Float16}(10.0, 2.0), 3, 1), 100, 100)) |> to_gpu
M = @inferred rand(curng, gpm);
@test eltype(M) == Float16

gpm = ProductDistribution(fill(KernelBinaryMixture(KernelExponential(2.0), KernelNormal(10.0, 2.0), 3, 1), 100, 100))
M = rand(rng, gpm);
maybe_histogram(flatten(M))
rand(pm) |> flatten |> maybe_histogram
rand(rng, gpm) |> flatten |> maybe_histogram
@inferred logdensityof(gpm, M)
# TODO product measure broken for BinaryMixture?
# @test logdensityof(gpm, M) ≈ logdensityof(pm, M)
@test logdensityof(gpm, M) ≈ logdensityof.((BinaryMixture(MeasureTheory.Exponential(λ=2.0), MeasureTheory.Normal(10.0, 2), 3, 1),), M) |> sum
@test logdensityof(gpm, M) isa Float64

M = rand(rng, gpm, 3);
@inferred logdensityof(gpm, M)
@test logdensityof(gpm, M) isa Float64

# VectorizedDistribution
pm = For(100, 100) do i, j
    MeasureTheory.Normal(i, j)
end;
gvm = VectorizedDistribution([KernelNormal{Float64}(i, j) for i = 1:100, j = 1:100])

M = @inferred rand(rng, gvm)
M = @inferred rand(curng, gvm)
gvm = to_gpu(gvm)
M = @inferred rand(curng, gvm)
gvm = VectorizedDistribution([KernelNormal{Float64}(i, j) for i = 1:100, j = 1:100])
M = @inferred rand(curng, gvm);
@test eltype(M) == Float64
gvm = VectorizedDistribution([KernelNormal{Float32}(i, j) for i = 1:100, j = 1:100])
M = @inferred rand(curng, gvm);
@test eltype(M) == Float32
gvm = VectorizedDistribution([KernelNormal{Float16}(i, j) for i = 1:100, j = 1:100])
M = @inferred rand(curng, gvm);
@test eltype(M) == Float16

gvm = VectorizedDistribution([KernelNormal{Float64}(i, j) for i = 1:100, j = 1:100])
M = rand(rng, gvm);
histogram(flatten(M))
rand(pm) |> flatten |> histogram
rand(curng, gvm, 10) |> flatten |> histogram
@inferred logdensityof(gvm, M)
@test logdensityof(gvm, M)[] |> sum ≈ logdensityof(pm, Array(M))
@test logdensityof(gvm, M) isa Vector{Float64}
@test logdensityof(gvm, M) |> size == (1,)

M = rand(rng, gvm, 3);
@inferred logdensityof(gvm, M)
@test logdensityof(gvm, M) isa Vector{Float64}
@test logdensityof(gvm, M) |> size == (3,)

# Broadcasting AbstractVectorizedKernel
gpm = ProductDistribution([KernelNormal{Float64}(i, j) for i = 1:100, j = 1:100]) |> to_gpu
gvm = VectorizedDistribution([KernelNormal{Float64}(i, j) for i = 1:100, j = 1:100]) |> to_gpu

M = @inferred rand(curng, gvm, 100, 5);
@inferred logdensityof(gpm, M)
@inferred logdensityof(gvm, M)
@test logdensityof(gpm, M) isa Real
@test logdensityof(gvm, M) |> size == (100, 5)
@test logdensityof(gpm, M) ≈ logdensityof(gvm, M) |> sum

# Method ambiguities
@inferred logdensityof(KernelExponential(Float64), 100)
@inferred logdensityof(KernelExponential(Float64), [100, 1])
@inferred logdensityof([KernelExponential(Float64), KernelExponential(Float64)], [100, 1])
@inferred logdensityof([KernelExponential(Float64), KernelExponential(Float64)], 100)
