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

# IndependentModel
s = @inferred rand(Random.default_rng(), abc_model)
@test variables(s).a isa Float16
@test variables(s).b isa Vector{Float32}
@test variables(s).c isa Vector{Float32}
@inferred logdensityof(IndependentModel((; a=a_model)), s)
@inferred logdensityof(ab_model, s)
@inferred logdensityof(ac_model, s)
@inferred logdensityof(bc_model, s)
@inferred logdensityof(abc_model, s)
s = @inferred rand(Random.default_rng(), abc_model, 3)
@test variables(s).a isa Vector{Float16}
@test variables(s).b isa Matrix{Float32}
@test variables(s).c isa Matrix{Float32}
@inferred logdensityof(ab_model, s)
@inferred logdensityof(ac_model, s)
@inferred logdensityof(bc_model, s)
@inferred logdensityof(abc_model, s)

# RngModel
xoshiro = Xoshiro()
rng_model = RngModel(abc_model, xoshiro)
# Same rng is used both times
Random.seed!(xoshiro, 42)
s1 = rand(Random.default_rng(), rng_model)
Random.seed!(xoshiro, 42)
s2 = rand(xoshiro, rng_model)
@test variables(s1).a == variables(s2).a
@test variables(s1).b == variables(s2).b
@test variables(s1).c == variables(s2).c
@test logdensityof(rng_model, s1) == logdensityof(rng_model, s2)
@test logdensityof(rng_model, s1) == logdensityof(abc_model, s2)

# ComposedModel
c_model = IndependentModel((; c=BroadcastedDistribution(KernelExponential, fill(2.0, 2))))
comp_model = @inferred ComposedModel(ab_model, bc_model, c_model)
s = @inferred rand(comp_model)
@test variables(s).a isa Float16
@test variables(s).b isa Vector{Float32}
@test variables(s).c isa Vector{Float64}
@test logdensityof(comp_model, s) == logdensityof(ab_model, s) + logdensityof(bc_model, s) + logdensityof(c_model, s)
