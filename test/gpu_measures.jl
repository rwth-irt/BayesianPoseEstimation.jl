# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 
using BenchmarkTools
using CUDA, MCMCDepth, MeasureTheory

# GpuNormal
gn = Normal(1.0, 2.0) |> gpu_measure
M = rand(gn, 100, 100)
rand!(gn, M)
logpdf.((gn,), M)
logpdf(gn, 1.0) ≈ logpdf(cpu_measure(gn), 1.0)

# GpuExponential
ge = Exponential(0.1) |> gpu_measure
M = rand(ge, 100, 100)
rand!(ge, M)
GE = CUDA.fill(ge, size(M))
logpdf.(GE, M)
logpdf(ge, 1.0) ≈ logpdf(cpu_measure(ge), 1.0)

# GpuUniformInterval
gu = UniformInterval(1.0, 10.0) |> gpu_measure
M = rand(gu, 100, 100)
rand!(gu, M)
MeasureTheory.logpdf.((gu,), M)
logpdf(gu, 0.5) == logpdf(UniformInterval(1.0, 10.0), 0.5)
logpdf(gu, 1.5) ≈ logpdf(cpu_measure(gu), 1.5)

# GpuBinaryMixture
gbm = BinaryMixture(Normal(1.0, 2.0), Normal(10.0, 0.1), 0.1, 0.9) |> gpu_measure
M = rand(gbm, 100, 100);
rand!(gbm, M);
logpdf.((gbm,), M)
logpdf(gbm, 1.0) ≈ logpdf(cpu_measure(gbm), 1.0)
