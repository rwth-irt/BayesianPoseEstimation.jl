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

a = SimpleNode(:a, rng, KernelUniform)
b = SimpleNode(:b, rng, KernelExponential)
c = SimpleNode(:c, rng, KernelNormal, (; a=a, b=b))
d = SimpleNode(:d, rng, KernelNormal, (; c=c, b=b))

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
@test prior_d == (; a=a, b=b, c=c)

# parent extraction
parent_a = parents(d, :a)
@test parent_a == (; c=c, d=d)
parent_c = parents(d, :c)
@test parent_c == (; d=d)
parent_ac = parents(d, a, c)
@test parent_ac == (; c=c, d=d)
parent_ba = parents(d, b, a)
@test parent_ba == (; c=c, d=d)

# evaluate deterministic nodes
fn(x, ::Any) = x
c = DeterministicNode(:c, fn, (; a=a, b=b))
d = SimpleNode(:d, rng, KernelNormal, (; c=c, b=b))
nt = evaluate(d, (; a=1, b=2.0f0, c=3.0f0, d=4.0f0, e=0.0f0))
@test nt == (; a=1, b=2.0f0, c=1.0f0, d=4.0f0, e=0.0f0)