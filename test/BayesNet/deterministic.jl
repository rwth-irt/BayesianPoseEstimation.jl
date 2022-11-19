# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# WARN Do not run this if you want Revise to work
include("../../src/MCMCDepth.jl")
using .MCMCDepth

using MCMCDepth
using Random
using Test

rng = Random.default_rng()

a = BroadcastedNode(:a, rng, KernelExponential, [1.0f0, 2, 3])
b = BroadcastedNode(:b, rng, KernelExponential, [1.0f0, 2])

fn(a, ::Any) = a
c = DeterministicNode(:c, fn, (; a=a, b=b))

nt = rand(c)
@test nt.c isa Array{Float32,1}
@test size(nt.c) == (3,)
ℓ = logdensityof(c, nt)
@test ℓ isa Float32
@test ℓ == logdensityof(a, nt) + logdensityof(b, nt)

nt = rand(c, 2)
@test nt.c isa Array{Float32,2}
@test size(nt.c) == (3, 2)
ℓ = logdensityof(c, nt)
@test ℓ isa Array{Float32,1}
@test ℓ == logden

# Test that evaluate only calls the DeterministicNode
nt = evaluate(c, (; a=1, b=2, c=3))
@test nt == (; a=1, b=2, c=1)
