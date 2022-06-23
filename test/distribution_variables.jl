# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using Bijectors
using CUDA
using DensityInterface
using Distributions
using MCMCDepth
using Random
using Test

# TODO Design decision: I will probably want to get rid of the Variables and operate on the raw arrays / scalars. This will allow me to get rid of logdensityof(::AbstractKernelDistribution, ::SpecializedVariable) which should be logdensityof(::Distribution, x::Any) or for unconstrained domains logdensityof(::TransformedDistribution, x::Any). Do I miss any features?

# TODO I would have to transform the VectorizedDistribution → does it work on the GPU?
# TODO wrong 
@inferred rand(Random.default_rng(), transformed(KernelExponential(Float16)), 3)
@inferred rand(Random.default_rng(), fill(transformed(KernelExponential(Float16)), 2), 3)

D = CUDA.fill(KernelExponential(Float16), 3)
T = transformed.(D)
X = @inferred rand(CUDA.default_rng(), D, 2)
Y = link.(D, X)
# TODO this is not expected
Y = @inferred rand(Random.default_rng(), Array(T), 2)
Y = @inferred rand(CUDA.default_rng(), T, 2)
Y = @inferred rand!(CUDA.default_rng(), D, Y)
Y = @inferred rand!(CUDA.default_rng(), T, Y)
X = invlink.(D, Y)

# TODO test rand! of transformed transformed_kernel_distributions.jl

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
@inferred logdensityof(a_model, a_mvar)
@test logdensityof(a_model, a_mvar) isa Float16

a_svar = SampleVariable(Random.default_rng(), a_model)
@inferred logdensityof(a_model, a_svar)
@test logdensityof(a_model, a_svar) isa Float16

# Bijector
a_dist = Exponential(Float16(1 / 2.0))
@test SampleVariable(a_mvar).value == bijector(a_dist)(a_mvar.value)
@test logdensityof(a_model, a_mvar) == logdensityof(a_dist, a_mvar.value)
@test logdensityof(a_model, a_svar) == logdensityof(transformed(a_dist), a_svar.value)

# Array
a_mvar = ModelVariable(CUDA.default_rng(), a_model, 3)
@inferred logdensityof(a_model, a_mvar)
@test logdensityof(a_model, a_mvar) isa AbstractVector{Float16}
@test logdensityof(a_model, a_mvar) |> size == (3,)

a_svar = SampleVariable(CUDA.default_rng(), a_model, 3)
@inferred logdensityof(a_model, a_svar)
@test logdensityof(a_model, a_svar) isa AbstractVector{Float16}
@test logdensityof(a_model, a_svar) |> size == (3,)

b_model = CUDA.fill(KernelExponential{Float16}(2.0), 3)

@inferred logdensityof(b_model, a_mvar)
@test logdensityof(b_model, a_mvar) isa AbstractVector{Float16}
@test logdensityof(b_model, a_mvar) |> size == (3,)

b_mvar = ModelVariable(CUDA.default_rng(), b_model)
@inferred logdensityof(b_model, b_mvar)
@test logdensityof(b_model, b_mvar) isa AbstractVector{Float16}
@test logdensityof(b_model, b_mvar) |> size == (3,)

b_svar = SampleVariable(CUDA.default_rng(), b_model, 3)
@inferred logdensityof(CuArray(b_model), b_svar)
@test logdensityof(b_model, b_svar) isa AbstractArray{Float16,2}
@test logdensityof(b_model, b_svar) |> size == (3, 3)

@test logdensityof(b_model, a_mvar) == logdensityof(a_model, a_mvar)
@test logdensityof(b_model, b_svar) == logdensityof(a_model, b_svar)

# Bijector
@test SampleVariable(a_mvar).value == bijector(a_dist)(a_mvar.value)
@test logdensityof(a_model, a_mvar) == logdensityof.(a_dist, a_mvar.value)
@test logdensityof(a_model, a_svar) == logdensityof.(transformed(a_dist), a_svar.value)

b_dists = fill(Exponential(Float16(1 / 2.0)), 3)
@test SampleVariable(b_mvar).value == bijector(first(b_dists))(b_mvar.value)
@test logdensityof(b_model, b_mvar) == logdensityof.(b_dists, Array(b_mvar.value))
@test logdensityof(b_model, b_svar) == logdensityof.(transformed.(b_dists), Array(b_svar.value))

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
@inferred logdensityof(a_model, a_mvar)
@test logdensityof(a_model, a_mvar) isa AbstractVector{Float16}
@test logdensityof(a_model, a_mvar) |> size == (4,)

a_svar = SampleVariable(CUDA.default_rng(), a_model, 4)
@inferred logdensityof(a_model, a_svar)
@test logdensityof(a_model, a_svar) isa AbstractVector{Float16}
@test logdensityof(a_model, a_svar) |> size == (4,)

b_model = ProductDistribution(fill(KernelExponential{Float16}(2.0), 3, 3))
b_mvar = ModelVariable(CUDA.default_rng(), b_model, 4)
@inferred logdensityof(b_model, b_mvar)
@test logdensityof(b_model, b_mvar) isa Float16

b_svar = SampleVariable(CUDA.default_rng(), b_model, 4)
@inferred logdensityof(b_model, b_svar)
@test logdensityof(b_model, b_svar) isa Float16

@test logdensityof(a_model, a_mvar) |> sum ≈ logdensityof(b_model, a_mvar)
@test logdensityof(a_model, a_svar) |> sum == logdensityof(b_model, a_svar)

# Bijector
a_dist = Exponential(Float16(1 / 2.0))
b_dist = Exponential(Float16(1 / 2.0))

@test logdensityof(a_model, a_mvar) == dropdims(sum(logdensityof.(a_dist, a_mvar.value), dims=(1, 2)); dims=(1, 2))
@test logdensityof(a_model, a_svar) == dropdims(sum(logdensityof.(transformed(a_dist), a_svar.value), dims=(1, 2)); dims=(1, 2))

@test logdensityof(b_model, b_mvar) == sum(logdensityof.(b_dist, b_mvar.value))
@test logdensityof(b_model, b_svar) == sum(logdensityof.(transformed(b_dist), b_svar.value))
