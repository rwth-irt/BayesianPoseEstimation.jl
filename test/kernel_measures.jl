# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.
using CUDA, MCMCDepth, MeasureTheory
using Plots
using Random
using Test

gn = Normal(10.0, 2.0) |> kernel_measure
M = rand(CUDA.default_rng(), gn, 100, 100)
M = rand(CUDA.default_rng(), fill(gn, 10), 10, 100)

# Correct device
@test rand(Random.default_rng(), KernelNormal(), 100, 100) isa Array
@test rand(CURAND.default_rng(), KernelNormal(), 100, 100) isa CuArray
@test rand(CUDA.default_rng(), KernelNormal(), 100, 100) isa CuArray

# KernelNormal
M = rand(CUDA.default_rng(), KernelNormal(Float64), 100, 100)
@test eltype(M) == Float64
M = rand(CUDA.default_rng(), KernelNormal(Float32), 100, 100)
@test eltype(M) == Float32
M = rand(CUDA.default_rng(), KernelNormal(), 100, 100)
@test eltype(M) == Float32
M = rand(CUDA.default_rng(), KernelNormal(Float16), 100, 100)
@test eltype(M) == Float16

gn = Normal(10.0, 2.0) |> kernel_measure
M = rand(CUDA.default_rng(), gn, 100, 100)
histogram(flatten(M))
histogram([rand(measure_theory(gn)) for _ in 1:100*100])
logpdf.((gn,), M)
@test logpdf(gn, 1.0) ≈ logpdf(measure_theory(gn), 1.0)

# KernelExponential
M = rand(CUDA.default_rng(), KernelExponential(Float64), 100, 100)
@test eltype(M) == Float64
M = rand(CUDA.default_rng(), KernelExponential(Float32), 100, 100)
@test eltype(M) == Float32
M = rand(CUDA.default_rng(), KernelExponential(), 100, 100)
@test eltype(M) == Float32
M = rand(CUDA.default_rng(), KernelExponential(Float16), 100, 100)
@test eltype(M) == Float16

ge = Exponential(0.1) |> kernel_measure
M = rand(CUDA.default_rng(), ge, 100, 100)
histogram(flatten(M))
histogram([rand(measure_theory(ge)) for _ in 1:100*100])
GE = CUDA.fill(ge, size(M))
logpdf.(GE, M)
@test logpdf(ge, 1.0) ≈ logpdf(measure_theory(ge), 1.0)

# KernelUniform
M = rand(CUDA.default_rng(), KernelUniform(Float64), 100, 100)
@test eltype(M) == Float64
M = rand(CUDA.default_rng(), KernelUniform(Float32), 100, 100)
@test eltype(M) == Float32
M = rand(CUDA.default_rng(), KernelUniform(), 100, 100)
@test eltype(M) == Float32
M = rand(CUDA.default_rng(), KernelUniform(Float16), 100, 100)
@test eltype(M) == Float16

gu = UniformInterval(5.0, 10.0) |> kernel_measure
M = rand(CUDA.default_rng(), gu, 100, 100)
histogram(flatten(M))
histogram([rand(measure_theory(gu)) for _ in 1:100*100])
MeasureTheory.logpdf.((gu,), M)
@test logpdf(gu, 0.5) == logpdf(measure_theory(gu), 0.5)
@test logpdf(gu, 1.5) ≈ logpdf(measure_theory(gu), 1.5)

# KernelCircularUniform
M = rand(CUDA.default_rng(), KernelCircularUniform(Float64), 100, 100)
@test eltype(M) == Float64
M = rand(CUDA.default_rng(), KernelCircularUniform(Float32), 100, 100)
@test eltype(M) == Float32
M = rand(CUDA.default_rng(), KernelCircularUniform(), 100, 100)
@test eltype(M) == Float32
M = rand(CUDA.default_rng(), KernelCircularUniform(Float16), 100, 100)
@test eltype(M) == Float16

gcu = CircularUniform() |> kernel_measure
M = rand(CUDA.default_rng(), gcu, 100, 100)
histogram(flatten(M))
histogram([rand(measure_theory(gcu)) for _ in 1:100*100])
MeasureTheory.logpdf.((gcu,), M)
@test logpdf(gcu, 0.5) ≈ logpdf(measure_theory(gcu), 0.5)
@test logpdf(gcu, 1.5) ≈ logpdf(measure_theory(gcu), 1.5)

# KernelBinaryMixture
bm = BinaryMixture(Exponential(2.0), Normal(10.0, 2), 3, 1)
M = rand(CUDA.default_rng(), kernel_measure(bm, Float64), 100, 100)
@test eltype(M) == Float64
M = rand(CUDA.default_rng(), kernel_measure(bm, Float32), 100, 100)
@test eltype(M) == Float32
M = rand(CUDA.default_rng(), kernel_measure(bm), 100, 100)
@test eltype(M) == Float32
M = rand(CUDA.default_rng(), kernel_measure(bm, Float16), 100, 100)
@test eltype(M) == Float16

gbm = kernel_measure(bm)
M = rand(CUDA.default_rng(), gbm, 100, 100);
histogram(flatten(M))
histogram([rand(measure_theory(gbm)) for _ in 1:100*100])
logpdf.((gbm,), M)
@test logpdf(gbm, 1.0) ≈ logpdf(measure_theory(gbm), 1.0)

# WARN Different measure types not supported only different parametrization of the same type
# KernelProduct
pm = For(100, 100) do i, j
    BinaryMixture(Exponential(2.0), Normal(10.0, 2), 3, 1)
end
gpm = KernelProduct(pm)
M = rand(Random.default_rng(), gpm)
M = rand(CUDA.default_rng(), gpm)
gpm = to_gpu(gpm)
M = rand(CUDA.default_rng(), gpm)
gpm = KernelProduct(pm, Float64) |> to_gpu
M = rand(CUDA.default_rng(), gpm);
@test eltype(M) == Float64
gpm = KernelProduct(pm, Float32) |> to_gpu
M = rand(CUDA.default_rng(), gpm);
@test eltype(M) == Float32
gpm = KernelProduct(pm) |> to_gpu
M = rand(CUDA.default_rng(), gpm);
@test eltype(M) == Float32
gpm = KernelProduct(pm, Float16) |> to_gpu
M = rand(CUDA.default_rng(), gpm);
@test eltype(M) == Float16

gpm = KernelProduct(pm) |> to_gpu
M = rand(CUDA.default_rng(), gpm);
histogram(flatten(M))
rand(measure_theory(gpm)) |> flatten |> histogram
rand(CUDA.default_rng(), gpm, 10) |> flatten |> histogram
logdensity(gpm, M)
logpdf(gpm, M)
@test logpdf(gpm, M) ≈ logpdf(measure_theory(gpm), Array(M))
@test logdensity(gpm, M) ≈ logdensity(measure_theory(gpm), Array(M))

# VectorizedMeasure
pm = For(100, 100) do i, j
    Normal(i, j)
end
gvm = VectorizedMeasure(pm)
@test kernel_measure(pm) isa VectorizedMeasure

M = rand(Random.default_rng(), gvm)
M = rand(CUDA.default_rng(), gvm)
gvm = to_gpu(gvm)
M = rand(CUDA.default_rng(), gvm)
gvm = VectorizedMeasure(pm, Float64) |> to_gpu
M = rand(CUDA.default_rng(), gvm);
@test eltype(M) == Float64
gvm = VectorizedMeasure(pm, Float32) |> to_gpu
M = rand(CUDA.default_rng(), gvm);
@test eltype(M) == Float32
gvm = VectorizedMeasure(pm) |> to_gpu
M = rand(CUDA.default_rng(), gvm);
@test eltype(M) == Float32
gvm = VectorizedMeasure(pm, Float16) |> to_gpu
M = rand(CUDA.default_rng(), gvm);
@test eltype(M) == Float16

gvm = VectorizedMeasure(pm, Float32) |> to_gpu
M = rand(CUDA.default_rng(), gvm)
histogram(flatten(M))
rand(measure_theory(gvm)) |> flatten |> histogram
rand(CUDA.default_rng(), gvm, 10) |> flatten |> histogram
logdensity(gvm, M)
logpdf(gvm, M)
@test logpdf(gvm, M) |> sum ≈ logpdf(pm, Array(M))
@test logdensity(gvm, M)[] |> sum ≈ logdensity(pm, Array(M))

# Broadcasting AbstractVectorizedMeasure
pm = For(10, 10) do i, j
    Normal(i, j)
end
gpm = KernelProduct(pm) |> to_gpu
gvm = VectorizedMeasure(pm) |> to_gpu

M = rand(CUDA.default_rng(), gvm, 100, 5);
@test logdensity(gpm, M) isa Real
@test logdensity(gvm, M) |> size == (100, 5)
@test logdensity(gpm, M) ≈ logdensity(gvm, M) |> sum
@test logpdf(gpm, M) isa Real
@test logpdf(gvm, M) |> size == (100, 5)
@test logpdf(gpm, M) ≈ logpdf(gvm, M) |> sum
