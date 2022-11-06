# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using MCMCDepth
using Random
using Test

rng = Random.default_rng()

a = BroadcastedNode(:a, KernelUniform, 0, fill(1.0f0, 3))
nt = @inferred rand(rng, a)
@test nt.a isa Array{Float32,1}
@test size(nt.a) == (3,)
ℓ = @inferred logdensityof(a, nt)
@test ℓ isa Float32
@test ℓ == 0
# Multiple times
nt = @inferred rand(rng, a, 2)
@test nt.a isa Array{Float32,2}
@test size(nt.a) == (3, 2)
ℓ = @inferred logdensityof(a, nt)
@test ℓ isa Array{Float32,1}
@test size(ℓ) == (2,)
@test ℓ == [0, 0]

b = BroadcastedNode(:b, KernelExponential, fill(1.0f0, 3, 4))
c = BroadcastedNode(:c, KernelNormal, (; a=a, b=b))
nt = rand(rng, c)
@test nt.c isa Array{Float32,2}
@test size(nt.c) == (3, 4)
ℓ = logdensityof(c, nt)
@test ℓ isa Float32
# WARN Multiple times requires matching sizes of children
b = BroadcastedNode(:b, KernelExponential, fill(1.0f0, 3))
c = BroadcastedNode(:c, KernelNormal, (; a=a, b=b))
nt = rand(rng, c, 2)
@test nt.a isa Array{Float32,2}
@test size(nt.a) == (3, 2)
ℓ = @inferred logdensityof(a, nt)
@test ℓ isa Array{Float32,1}
@test size(ℓ) == (2,)
@test ℓ == [0, 0]

# TODO d = VariableNode(:d, KernelNormal, (; c=c, b=b))

# TEST NOT done yet
@test ℓ == logdensityof(KernelUniform(), nt.a) + logdensityof(KernelExponential(), nt.b) + logdensityof(KernelNormal(nt.a, nt.b), nt.c) + logdensityof(KernelNormal(nt.c, nt.b), nt.d)
bij = bijector(c_broad)
@test bij isa NamedTuple{(:a, :b, :c)}
@test values(bij) == (bijector(KernelUniform()), bijector(KernelExponential()), bijector(KernelNormal()), bijector(KernelNormal()))
