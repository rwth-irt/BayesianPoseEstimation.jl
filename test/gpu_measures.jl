# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.
using CUDA, MCMCDepth, MeasureTheory
using Plots
using Random
using Test

gn = Normal(10.0, 2.0) |> gpu_measure
M = rand(CUDA.default_rng(), gn, 100, 100)
rand!(gn, M)

# Correct device
@test rand(Random.default_rng(), GpuNormal(), 100, 100) isa Array
@test rand(CURAND.default_rng(), GpuNormal(), 100, 100) isa CuArray
@test rand(CUDA.default_rng(), GpuNormal(), 100, 100) isa CuArray

# GpuNormal
M = rand(CUDA.default_rng(), GpuNormal(Float64), 100, 100)
@test eltype(M) == Float64
M = rand(CUDA.default_rng(), GpuNormal(Float32), 100, 100)
@test eltype(M) == Float32
M = rand(CUDA.default_rng(), GpuNormal(), 100, 100)
@test eltype(M) == Float32
M = rand(CUDA.default_rng(), GpuNormal(Float16), 100, 100)
@test eltype(M) == Float16

gn = Normal(10.0, 2.0) |> gpu_measure
M = rand(CUDA.default_rng(), gn, 100, 100)
histogram(flatten(M))
rand!(gn, M)
histogram(flatten(M))
histogram([rand(cpu_measure(gn)) for _ in 1:100*100])
logpdf.((gn,), M)
@test logpdf(gn, 1.0) ≈ logpdf(cpu_measure(gn), 1.0)

# GpuExponential
M = rand(CUDA.default_rng(), GpuExponential(Float64), 100, 100)
@test eltype(M) == Float64
M = rand(CUDA.default_rng(), GpuExponential(Float32), 100, 100)
@test eltype(M) == Float32
M = rand(CUDA.default_rng(), GpuExponential(), 100, 100)
@test eltype(M) == Float32
M = rand(CUDA.default_rng(), GpuExponential(Float16), 100, 100)
@test eltype(M) == Float16

ge = Exponential(0.1) |> gpu_measure
M = rand(CUDA.default_rng(), ge, 100, 100)
histogram(flatten(M))
rand!(ge, M)
histogram(flatten(M))
histogram([rand(cpu_measure(ge)) for _ in 1:100*100])
GE = CUDA.fill(ge, size(M))
logpdf.(GE, M)
@test logpdf(ge, 1.0) ≈ logpdf(cpu_measure(ge), 1.0)

# GpuUniformInterval
M = rand(CUDA.default_rng(), GpuUniformInterval(Float64), 100, 100)
@test eltype(M) == Float64
M = rand(CUDA.default_rng(), GpuUniformInterval(Float32), 100, 100)
@test eltype(M) == Float32
M = rand(CUDA.default_rng(), GpuUniformInterval(), 100, 100)
@test eltype(M) == Float32
M = rand(CUDA.default_rng(), GpuUniformInterval(Float16), 100, 100)
@test eltype(M) == Float16

gu = UniformInterval(1.0, 10.0) |> gpu_measure
M = rand(CUDA.default_rng(), gu, 100, 100)
histogram(flatten(M))
rand!(gu, M)
histogram(flatten(M))
MeasureTheory.logpdf.((gu,), M)
@test logpdf(gu, 0.5) == logpdf(cpu_measure(gu), 0.5)
@test logpdf(gu, 1.5) ≈ logpdf(cpu_measure(gu), 1.5)

# GpuCircularUniform
M = rand(CUDA.default_rng(), GpuCircularUniform(Float64), 100, 100)
@test eltype(M) == Float64
M = rand(CUDA.default_rng(), GpuCircularUniform(Float32), 100, 100)
@test eltype(M) == Float32
M = rand(CUDA.default_rng(), GpuCircularUniform(), 100, 100)
@test eltype(M) == Float32
M = rand(CUDA.default_rng(), GpuCircularUniform(Float16), 100, 100)
@test eltype(M) == Float16

gcu = CircularUniform() |> gpu_measure
M = rand(CUDA.default_rng(), gcu, 100, 100)
histogram(flatten(M))
rand!(gcu, M)
histogram(flatten(M))
MeasureTheory.logpdf.((gcu,), M)
@test logpdf(gcu, 0.5) ≈ logpdf(cpu_measure(gcu), 0.5)
@test logpdf(gcu, 1.5) ≈ logpdf(cpu_measure(gcu), 1.5)

# GpuBinaryMixture
bm = BinaryMixture(Exponential(2.0), Normal(10.0, 2), 3, 1)
M = rand(CUDA.default_rng(), gpu_measure(bm, Float64), 100, 100)
@test eltype(M) == Float64
M = rand(CUDA.default_rng(), gpu_measure(bm, Float32), 100, 100)
@test eltype(M) == Float32
M = rand(CUDA.default_rng(), gpu_measure(bm), 100, 100)
@test eltype(M) == Float32
M = rand(CUDA.default_rng(), gpu_measure(bm, Float16), 100, 100)
@test eltype(M) == Float16

gbm = gpu_measure(bm)
M = rand(CUDA.default_rng(), gbm, 100, 100);
histogram(flatten(M))
rand!(gbm, M);
histogram(flatten(M))
logpdf.((gbm,), M)
@test logpdf(gbm, 1.0) ≈ logpdf(cpu_measure(gbm), 1.0)

# WARN Different measure types not supported only different parametrization of the same type
# GpuProductMeasure
pm = For(10, 10, 100) do i, j, k
    Normal(i, j)
end
gpm = gpu_measure(pm)
M = rand(Random.default_rng(), gpm)
M = rand(CUDA.default_rng(), gpm)
gpm = to_gpu(gpm)
M = rand(CUDA.default_rng(), gpm)
gpm = gpu_measure(pm, Float64) |> to_gpu
M = rand(CUDA.default_rng(), gpm);
@test eltype(M) == Float64
gpm = gpu_measure(pm, Float32) |> to_gpu
M = rand(CUDA.default_rng(), gpm);
@test eltype(M) == Float32
gpm = gpu_measure(pm) |> to_gpu
M = rand(CUDA.default_rng(), gpm);
@test eltype(M) == Float32
gpm = gpu_measure(pm, Float16) |> to_gpu
M = rand(CUDA.default_rng(), gpm);
@test eltype(M) == Float16

gpm = gpu_measure(pm) |> to_gpu
M = rand(CUDA.default_rng(), gpm);
histogram(flatten(M))
rand!(gpm, M);
histogram(flatten(M))
logdensity(gpm, M)
logpdf(gpm, M)
@test logpdf(gpm, M) ≈ logpdf(cpu_measure(gpm), Array(M))
@test logdensity(gpm, M) ≈ logdensity(cpu_measure(gpm), Array(M))

# GpuVectorizedMeasure
pm = For(10, 10, 100) do i, j, k
    Normal(i, j)
end
gvm = GpuVectorizedMeasure(pm)

M = rand(Random.default_rng(), gvm)
M = rand(CUDA.default_rng(), gvm)
gvm = to_gpu(gvm)
M = rand(CUDA.default_rng(), gvm)
gvm = GpuVectorizedMeasure(pm, Float64) |> to_gpu
M = rand(CUDA.default_rng(), gvm);
@test eltype(M) == Float64
gvm = GpuVectorizedMeasure(pm, Float32) |> to_gpu
M = rand(CUDA.default_rng(), gvm);
@test eltype(M) == Float32
gvm = GpuVectorizedMeasure(pm) |> to_gpu
M = rand(CUDA.default_rng(), gvm);
@test eltype(M) == Float32
gvm = GpuVectorizedMeasure(pm, Float16) |> to_gpu
M = rand(CUDA.default_rng(), gvm);
@test eltype(M) == Float16

gvm = GpuVectorizedMeasure(pm, Float32) |> to_gpu
M = rand(CUDA.default_rng(), gvm);
histogram(flatten(M))
rand!(gvm, M);
histogram(flatten(M))
logdensity(gvm, M)
logpdf(gvm, M)
@test logpdf(gvm, M) |> sum ≈ logpdf(pm, Array(M))
@test logdensity(gvm, M) |> sum ≈ logdensity(pm, Array(M))

# Broadcasting AbstractVectorizedMeasure

pm = For(10, 10) do i, j
    Normal(i, j)
end
gpm = GpuProductMeasure(pm)
gvm = GpuVectorizedMeasure(pm) |> to_gpu
large_pm = For(10, 10, 100) do i, j, k
    Normal(i, j)
end
large_gpm = GpuProductMeasure(large_pm)
M = rand(CUDA.default_rng(), large_gpm);
@test logdensity(gpm, M) isa Real
@test logdensity(gvm, M) |> length == 100
@test logdensity(gpm, M) == logdensity(gvm, M) |> sum
@test logpdf(gpm, M) isa Real
@test logpdf(gvm, M) |> length == 100
@test logpdf(gpm, M) ≈ logpdf(gvm, M) |> sum
