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

a = SimpleNode(:a, rng, KernelUniform())
b = SimpleNode(:b, rng, KernelExponential())
c = SimpleNode(:c, (; a=a, b=b), rng, KernelNormal)
d = SimpleNode(:d, (; c=c, b=b), rng, KernelNormal)

nt = rand(d, (; a=1))
@test nt.a == 1
nt = rand(d)
@test nt.a != 1
@test nt.d isa Float32
ℓ = logdensityof(d, nt)
@test ℓ isa Float32
@test ℓ == logdensityof(KernelUniform(), nt.a) + logdensityof(KernelExponential(), nt.b) + logdensityof(KernelNormal(nt.a, nt.b), nt.c) + logdensityof(KernelNormal(nt.c, nt.b), nt.d)
bij = bijector(d)
@test bij isa NamedTuple{(:a, :b, :c, :d)}
@test values(bij) == (bijector(KernelUniform()), bijector(KernelExponential()), bijector(KernelNormal()), bijector(KernelNormal()))

# multiple samples not supported
@test_throws MethodError rand(d, 2)

# prior extraction
prior_d = prior(d)
@test prior_d == (; a=a, b=b)

# parent extraction
parent_a = MCMCDepth.parents(:a, d)
@test parent_a == (; c=c, d=d)