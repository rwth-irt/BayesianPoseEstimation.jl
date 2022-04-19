# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.
using CUDA, MCMCDepth, MeasureTheory
using Plots
using Random
using Test

gn = Normal(10.0, 2.0) |> kernel_distribution
# Yup 42 is bad style, like this emoji 🤣
curng = CUDA.RNG(42)
rng = Random.default_rng(42)
M = rand(curng, gn, 100, 100)
M = rand(curng, CUDA.fill(gn, 10), 10, 100)
M = rand(curng, fill(gn, 10), 10, 100)

# Correct device
@test rand(curng, KernelNormal(), 100, 100) isa CuArray
@test rand(rng, KernelNormal(), 100, 100) isa Array
# @test rand(CURAND.default_rng(), KernelNormal(), 100, 100) isa CuArray

# TODO more @inferred
# KernelNormal
M = @inferred rand(curng, KernelNormal(Float64), 100, 100)
@test eltype(M) == Float64
M = @inferred rand(curng, KernelNormal(Float32), 100, 100)
@test eltype(M) == Float32
M = @inferred rand(curng, KernelNormal(), 100, 100)
@test eltype(M) == Float32
M = @inferred rand(curng, KernelNormal(Float16), 100, 100)
@test eltype(M) == Float16

gn = Normal(10.0, 2.0) |> kernel_distribution
M = rand(curng, gn, 100, 100)
histogram(flatten(M))
histogram([rand(measure_theory(gn)) for _ in 1:100*100])
@inferred logdensityof(gn, M)
@test logdensityof(gn, 1.0) ≈ logdensityof(measure_theory(gn), 1.0)

# KernelExponential
M = @inferred rand(curng, KernelExponential(Float64), 100, 100)
@test eltype(M) == Float64
M = @inferred rand(curng, KernelExponential(Float32), 100, 100)
@test eltype(M) == Float32
M = @inferred rand(curng, KernelExponential(), 100, 100)
@test eltype(M) == Float32
M = @inferred rand(curng, KernelExponential(Float16), 100, 100)
@test eltype(M) == Float16

ge = Exponential(0.1) |> kernel_distribution
M = rand(curng, ge, 100, 100)
histogram(flatten(M))
histogram([rand(measure_theory(ge)) for _ in 1:100*100])
@inferred logdensityof(ge, M)
@test logdensityof(ge, 1.0) ≈ logdensityof(measure_theory(ge), 1.0)

# KernelUniform
M = @inferred rand(curng, KernelUniform(Float64), 100, 100)
@test eltype(M) == Float64
M = @inferred rand(curng, KernelUniform(Float32), 100, 100)
@test eltype(M) == Float32
M = @inferred rand(curng, KernelUniform(), 100, 100)
@test eltype(M) == Float32
M = @inferred rand(curng, KernelUniform(Float16), 100, 100)
@test eltype(M) == Float16

gu = UniformInterval(5.0, 10.0) |> kernel_distribution
M = rand(curng, gu, 100, 100)
histogram(flatten(M))
histogram([rand(measure_theory(gu)) for _ in 1:100*100])
@inferred logdensityof(gu, M)
@test logdensityof(gu, 0.5) == logdensityof(measure_theory(gu), 0.5)
@test logdensityof(gu, 1.5) ≈ logdensityof(measure_theory(gu), 1.5)

# KernelCircularUniform
M = @inferred rand(curng, KernelCircularUniform(Float64), 100, 100)
@test eltype(M) == Float64
M = @inferred rand(curng, KernelCircularUniform(Float32), 100, 100)
@test eltype(M) == Float32
M = @inferred rand(curng, KernelCircularUniform(), 100, 100)
@test eltype(M) == Float32
M = @inferred rand(curng, KernelCircularUniform(Float16), 100, 100)
@test eltype(M) == Float16

gcu = CircularUniform() |> kernel_distribution
M = rand(curng, gcu, 100, 100)
histogram(flatten(M))
histogram([rand(measure_theory(gcu)) for _ in 1:100*100])
@inferred logdensityof(gcu, M)
@test logdensityof(gcu, 0.5) ≈ logdensityof(measure_theory(gcu), 0.5)
@test logdensityof(gcu, 1.5) ≈ logdensityof(measure_theory(gcu), 1.5)

# KernelBinaryMixture
bm = BinaryMixture(Exponential(2.0), Normal(10.0, 2), 3, 1)
M = @inferred rand(curng, kernel_distribution(bm, Float64), 100, 100)
@test eltype(M) == Float64
M = @inferred rand(curng, kernel_distribution(bm, Float32), 100, 100)
@test eltype(M) == Float32
M = @inferred rand(curng, kernel_distribution(bm), 100, 100)
@test eltype(M) == Float32
M = @inferred rand(curng, kernel_distribution(bm, Float16), 100, 100)
@test eltype(M) == Float16

gbm = kernel_distribution(bm)
M = rand(curng, gbm, 100, 100);
histogram(flatten(M))
histogram([rand(measure_theory(gbm)) for _ in 1:100*100])
@inferred logdensityof(gbm, M)
@test logdensityof(gbm, 1.0) ≈ logdensityof(measure_theory(gbm), 1.0)

# WARN Different measure types not supported only different parametrization of the same type
# KernelProduct
pm = For(100, 100) do i, j
    BinaryMixture(Exponential(2.0), Normal(10.0, 2), 3, 1)
end;
gpm = KernelProduct(pm)
M = @inferred rand(rng, gpm)
M = @inferred rand(curng, gpm)
gpm = to_gpu(gpm)
M = @inferred rand(curng, gpm)
gpm = KernelProduct(pm, Float64) |> to_gpu
M = @inferred rand(curng, gpm);
@test eltype(M) == Float64
gpm = KernelProduct(pm, Float32) |> to_gpu
M = @inferred rand(curng, gpm);
@test eltype(M) == Float32
gpm = KernelProduct(pm) |> to_gpu
M = @inferred rand(curng, gpm);
@test eltype(M) == Float32
gpm = KernelProduct(pm, Float16) |> to_gpu
M = @inferred rand(curng, gpm);
@test eltype(M) == Float16

gpm = KernelProduct(pm) |> to_gpu
M = rand(curng, gpm);
histogram(flatten(M))
rand(measure_theory(gpm)) |> flatten |> histogram
rand(curng, gpm, 10) |> flatten |> histogram
@inferred logdensityof(gpm, M)
@test logdensityof(gpm, M) ≈ logdensityof(measure_theory(gpm), Array(M))

# KernelVectorized
pm = For(100, 100) do i, j
    Normal(i, j)
end;
gvm = KernelVectorized(pm)
@test kernel_distribution(pm) isa KernelVectorized

M = @inferred rand(rng, gvm)
M = @inferred rand(curng, gvm)
gvm = to_gpu(gvm)
M = @inferred rand(curng, gvm)
gvm = KernelVectorized(pm, Float64) |> to_gpu
M = @inferred rand(curng, gvm);
@test eltype(M) == Float64
gvm = KernelVectorized(pm, Float32) |> to_gpu
M = @inferred rand(curng, gvm);
@test eltype(M) == Float32
gvm = KernelVectorized(pm) |> to_gpu
M = @inferred rand(curng, gvm);
@test eltype(M) == Float32
gvm = KernelVectorized(pm, Float16) |> to_gpu
M = @inferred rand(curng, gvm);
@test eltype(M) == Float16

gvm = KernelVectorized(pm, Float32) |> to_gpu
M = rand(curng, gvm)
histogram(flatten(M))
rand(measure_theory(gvm)) |> flatten |> histogram
rand(curng, gvm, 10) |> flatten |> histogram
@inferred logdensityof(gvm, M)
@test logdensityof(gvm, M)[] |> sum ≈ logdensityof(pm, Array(M))

# Broadcasting AbstractVectorizedKernel
pm = For(10, 10) do i, j
    Normal(i, j)
end
gpm = KernelProduct(pm) |> to_gpu
gvm = KernelVectorized(pm) |> to_gpu

M = @inferred rand(curng, gvm, 100, 5);
@inferred logdensityof(gpm, M)
@inferred logdensityof(gvm, M)
@test logdensityof(gpm, M) isa Real
@test logdensityof(gvm, M) |> size == (100, 5)
@test logdensityof(gpm, M) ≈ logdensityof(gvm, M) |> sum
