# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.
using CUDA, MCMCDepth, MeasureTheory
using Plots
using Test

# GpuNormal
gn = Normal(10.0, 2.0) |> gpu_measure
M = rand(CURAND.default_rng(), Float64, gn, 100, 100)
@test eltype(M) == Float64
M = rand(Float32, gn, 100, 100)
@test eltype(M) == Float32
M = rand(gn, 100, 100)
@test eltype(M) == Float32
histogram(flatten(M))
rand!(gn, M)
histogram(flatten(M))
logpdf.((gn,), M)
@test logpdf(gn, 1.0) ≈ logpdf(cpu_measure(gn), 1.0)

# GpuExponential
ge = Exponential(0.1) |> gpu_measure
M = rand(ge, 100, 100)
histogram(flatten(M))
rand!(ge, M)
histogram(flatten(M))
GE = CUDA.fill(ge, size(M))
logpdf.(GE, M)
@test logpdf(ge, 1.0) ≈ logpdf(cpu_measure(ge), 1.0)

# GpuUniformInterval
gu = UniformInterval(1.0, 10.0) |> gpu_measure
M = rand(gu, 100, 100)
histogram(flatten(M))
rand!(gu, M)
histogram(flatten(M))
MeasureTheory.logpdf.((gu,), M)
@test logpdf(gu, 0.5) == logpdf(cpu_measure(gu), 0.5)
@test logpdf(gu, 1.5) ≈ logpdf(cpu_measure(gu), 1.5)

# GpuCircularUniform
gcu = CircularUniform() |> gpu_measure
M = rand(gcu, 100, 100)
histogram(flatten(M))
rand!(gcu, M)
histogram(flatten(M))
MeasureTheory.logpdf.((gcu,), M)
@test logpdf(gcu, 0.5) == logpdf(cpu_measure(gcu), 0.5)
@test logpdf(gcu, 1.5) ≈ logpdf(cpu_measure(gcu), 1.5)

# GpuBinaryMixture
gbm = BinaryMixture(Normal(1.0, 2.0), Normal(10.0, 0.1), 0.1, 0.9) |> gpu_measure
M = rand(gbm, 100, 100);
histogram(flatten(M))
rand!(gbm, M);
histogram(flatten(M))
logpdf.((gbm,), M)
@test logpdf(gbm, 1.0) ≈ logpdf(cpu_measure(gbm), 1.0)

# GpuProductMeasure
pm = For(10, 10, 100) do i, j, k
    Normal()
end
gpm = pm |> gpu_measure
M = rand(CURAND.default_rng(), Float64, gpm);
@test eltype(M) == Float64
M = rand(Float32, gpm);
@test eltype(M) == Float32
M = rand(gpm);
@test eltype(M) == Float32
histogram(flatten(M))
rand!(gpm, M);
histogram(flatten(M))
logdensity(gpm, M)
logpdf(gpm, M)
@test logpdf(gpm, M) ≈ logpdf(cpu_measure(gpm), Array(M))
@test logdensity(gpm, M) ≈ logdensity(cpu_measure(gpm), Array(M))


# GpuVectorizedMeasure
vm = VectorizedMeasure(Normal(1.0, 2.0), 10, 10, 100)
gvm = vm |> gpu_measure
M = rand(CURAND.default_rng(), Float64, gvm);
@test eltype(M) == Float64
M = rand(Float32, gvm);
@test eltype(M) == Float32
M = rand(gvm);
@test eltype(M) == Float32
histogram(flatten(M))
rand!(gvm, M);
histogram(flatten(M))
logdensity(gvm, M)
logpdf(gvm, M)
@test logpdf(gvm, M) |> Array ≈ logpdf(cpu_measure(gvm), Array(M))
@test logdensity(gvm, M) |> Array ≈ logdensity(cpu_measure(gvm), Array(M))