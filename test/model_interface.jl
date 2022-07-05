# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 


# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using Distributions
using MCMCDepth
using Random
using Test

a_model = KernelExponential(Float16(2.0))
b_model = BroadcastedDistribution(Exponential, [2.0f0, 1.0f0, 0.5f0])
c_model = BroadcastedDistribution(KernelExponential, fill(2.0f0, 2))

ab_model = IndependentModel((; a=a_model, b=b_model))
ac_model = IndependentModel((; a=a_model, c=c_model))
bc_model = IndependentModel((; a=a_model, c=c_model))
abc_model = IndependentModel((; a=a_model, b=b_model, c=c_model))

# TODO test expected type
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
