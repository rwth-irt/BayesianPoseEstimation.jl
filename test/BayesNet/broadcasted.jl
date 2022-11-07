# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# WARN Do not run this if you want Revise to work
include("../../src/MCMCDepth.jl")
using .MCMCDepth

using Base: materialize
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
@test nt.c isa Array{Float32,2}
@test size(nt.c) == (3, 2)
ℓ = logdensityof(c, nt)
@test ℓ isa Array{Float32,1}
@test size(ℓ) == (2,)

# Test logdensityof calculation
d = BroadcastedNode(:d, KernelNormal, (; c=c, b=b))
nt = rand(rng, d)
ℓ = logdensityof(d, nt)
@test ℓ == sum(logdensityof.(KernelUniform(), nt.a) + logdensityof.(KernelExponential(), nt.b) + logdensityof.(KernelNormal.(nt.a, nt.b), nt.c) + logdensityof.(KernelNormal.(nt.c, nt.b), nt.d))

# Test bijectors
bij = bijector(d)
@test bij isa NamedTuple{(:a, :b, :c, :d)}
a_bij = bijector(ProductBroadcastedDistribution(KernelUniform, 0, fill(1.0f0, 3)))
@test bij.a.dims == a_bij.dims
@test materialize(bij.a.bijectors) == materialize(a_bij.bijectors)
b_bij = bijector(ProductBroadcastedDistribution(KernelExponential, fill(1.0f0, 3)))
@test bij.b.dims == b_bij.dims
@test materialize(bij.b.bijectors) == materialize(b_bij.bijectors)
c_bij = bijector(ProductBroadcastedDistribution(KernelNormal, 0, fill(1.0f0, 3)))
@test bij.c.dims == c_bij.dims
@test materialize(bij.c.bijectors) == materialize(c_bij.bijectors)
d_bij = bijector(ProductBroadcastedDistribution(KernelNormal, 0, fill(1.0f0, 3)))
@test bij.d.dims == d_bij.dims
@test materialize(bij.d.bijectors) == materialize(d_bij.bijectors)
