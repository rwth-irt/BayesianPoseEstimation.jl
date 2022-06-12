# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 


# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using MCMCDepth
using Random
using Test

a_model = KernelExponential(2.0)
# Float32
b_model = ProductDistribution([KernelExponential(2.0f0), KernelExponential(1.0f0), KernelExponential(0.5f0)])
c_model = VectorizedDistribution(fill(KernelExponential(2.0), 2))

ab_model = IndependentModel((; a=a_model, b=b_model))
ac_model = IndependentModel((; a=a_model, c=c_model))
bc_model = IndependentModel((; a=a_model, c=c_model))
abc_model = IndependentModel((; a=a_model, b=b_model, c=c_model))

s = @inferred rand(Random.default_rng(), abc_model)
@inferred logdensityof(IndependentModel((; a=a_model)), s)
@inferred logdensityof(ab_model, s)
@inferred logdensityof(ac_model, s)
@inferred logdensityof(bc_model, s)
@inferred logdensityof(abc_model, s)
s = @inferred rand(Random.default_rng(), abc_model, 3)
@inferred logdensityof(ab_model, s)
@inferred logdensityof(ac_model, s)
@inferred logdensityof(bc_model, s)
@inferred logdensityof(abc_model, s)
