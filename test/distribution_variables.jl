# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using CUDA
using DensityInterface
using MCMCDepth
using Random
using Test

# KernelDistribution Random 

@inferred ModelVariable(KernelExponential())
@inferred SampleVariable(KernelExponential())
@inferred ModelVariable(Random.MersenneTwister(), KernelExponential())
@inferred SampleVariable(Random.MersenneTwister(), KernelExponential())

@inferred ModelVariable(KernelExponential(), 2)
@inferred SampleVariable(KernelExponential(), 2)
@inferred ModelVariable(fill(KernelExponential(), 2))
@inferred SampleVariable(fill(KernelExponential(), 2))
@test size(model_value(ModelVariable(KernelExponential(), 2))) == size(model_value(ModelVariable(fill(KernelExponential(), 2))))
# WARN CUDA does not support scalars → wrap in array
@inferred ModelVariable(CUDA.default_rng(), KernelExponential(), 2)
@inferred SampleVariable(CUDA.default_rng(), KernelExponential(), 2)

# KernelDistribution DensityInterface

# Scalar (CUDA only supports Arrays)
a_model = KernelExponential{Float16}(2.0)

a_mvar = ModelVariable(Random.default_rng(), a_model)
@inferred DensityInterface.logdensityof(a_model, a_mvar)
@test DensityInterface.logdensityof(a_model, a_mvar) isa Float16

a_svar = SampleVariable(Random.default_rng(), a_model)
@inferred DensityInterface.logdensityof(a_model, a_svar)
@test DensityInterface.logdensityof(a_model, a_svar) isa Float16

# Array
a_mvar = ModelVariable(CUDA.default_rng(), a_model, 3)
@inferred DensityInterface.logdensityof(a_model, a_mvar)
@test DensityInterface.logdensityof(a_model, a_mvar) isa AbstractVector{Float16}
@test DensityInterface.logdensityof(a_model, a_mvar) |> size == (3,)

a_svar = SampleVariable(CUDA.default_rng(), a_model, 3)
@inferred DensityInterface.logdensityof(a_model, a_svar)
@test DensityInterface.logdensityof(a_model, a_svar) isa AbstractVector{Float16}
@test DensityInterface.logdensityof(a_model, a_svar) |> size == (3,)

b_model = fill(KernelExponential{Float16}(2.0), 3)
@inferred DensityInterface.logdensityof(b_model, a_mvar)
@test DensityInterface.logdensityof(b_model, a_mvar) isa AbstractVector{Float16}
@test DensityInterface.logdensityof(b_model, a_mvar) |> size == (3,)

b_svar = SampleVariable(CUDA.default_rng(), b_model, 3)
@inferred DensityInterface.logdensityof(CuArray(b_model), b_svar)
@test DensityInterface.logdensityof(b_model, b_svar) isa AbstractVector{Float16}
@test DensityInterface.logdensityof(b_model, b_svar) |> size == (3,)

@test DensityInterface.logdensityof(b_model, a_mvar) == DensityInterface.logdensityof(a_model, a_mvar)
@test DensityInterface.logdensityof(b_model, b_svar) == DensityInterface.logdensityof(a_model, b_svar)

# VectorizedDistribution Random 

@inferred ModelVariable(VectorizedDistribution(fill(KernelExponential(), 3, 3)))
@inferred SampleVariable(VectorizedDistribution(fill(KernelExponential(), 3, 3)))
@inferred ModelVariable(Random.MersenneTwister(), VectorizedDistribution(fill(KernelExponential(), 3, 3)))
@inferred SampleVariable(Random.MersenneTwister(), VectorizedDistribution(fill(KernelExponential(), 3, 3)))

@inferred ModelVariable(VectorizedDistribution(fill(KernelExponential(), 3, 3)), 2)
@inferred SampleVariable(VectorizedDistribution(fill(KernelExponential(), 3, 3)), 2)
@inferred ModelVariable(Random.MersenneTwister(), VectorizedDistribution(fill(KernelExponential(), 3, 3)), 2)
@inferred SampleVariable(Random.MersenneTwister(), VectorizedDistribution(fill(KernelExponential(), 3, 3)), 2)

@inferred ModelVariable(CUDA.default_rng(), VectorizedDistribution(fill(KernelExponential(), 3, 3)))
@inferred SampleVariable(CUDA.default_rng(), VectorizedDistribution(fill(KernelExponential(), 3, 3)))
@inferred ModelVariable(CUDA.default_rng(), VectorizedDistribution(fill(KernelExponential(), 3, 3)), 2)
@inferred SampleVariable(CUDA.default_rng(), VectorizedDistribution(fill(KernelExponential(), 3, 3)), 2)

# VectorizedDistribution DensityInterface

a_model = VectorizedDistribution(fill(KernelExponential{Float16}(2.0), 3, 3))
a_mvar = ModelVariable(CUDA.default_rng(), a_model, 4)
@inferred DensityInterface.logdensityof(a_model, a_mvar)
@test DensityInterface.logdensityof(a_model, a_mvar) isa AbstractVector{Float16}
@test DensityInterface.logdensityof(a_model, a_mvar) |> size == (4,)

a_svar = SampleVariable(CUDA.default_rng(), a_model, 4)
@inferred DensityInterface.logdensityof(a_model, a_svar)
@test DensityInterface.logdensityof(a_model, a_svar) isa AbstractVector{Float16}
@test DensityInterface.logdensityof(a_model, a_svar) |> size == (4,)

b_model = ProductDistribution(fill(KernelExponential{Float16}(2.0), 3, 3))
a_mvar = ModelVariable(CUDA.default_rng(), b_model, 4)
@inferred DensityInterface.logdensityof(b_model, a_mvar)
@test DensityInterface.logdensityof(b_model, a_mvar) isa Float16

a_svar = SampleVariable(CUDA.default_rng(), b_model, 4)
@inferred DensityInterface.logdensityof(b_model, a_svar)
@test DensityInterface.logdensityof(b_model, a_svar) isa Float16

@test DensityInterface.logdensityof(a_model, a_mvar) |> sum == DensityInterface.logdensityof(b_model, a_mvar)
@test DensityInterface.logdensityof(a_model, a_svar) |> sum ≈ DensityInterface.logdensityof(b_model, a_svar)
