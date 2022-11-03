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

a = VariableNode(:a, KernelUniform())
b = VariableNode(:b, KernelExponential())
c = VariableNode(:c, KernelNormal, (; a=a, b=b))
d = VariableNode(:d, KernelNormal, (; c=c, b=b))

nt = rand(rng, d)
@test nt.d isa Float32
ℓ = logdensityof(d, nt)
@test ℓ isa Float32
@test ℓ == logdensityof(KernelUniform(), nt.a) + logdensityof(KernelExponential(), nt.b) + logdensityof(KernelNormal(nt.a, nt.b), nt.c) + logdensityof(KernelNormal(nt.c, nt.b), nt.d)
bij = bijector(d)
@test bij isa NamedTuple{(:a, :b, :c, :d)}
@test values(bij) == (bijector(KernelUniform()), bijector(KernelExponential()), bijector(KernelNormal()), bijector(KernelNormal()))

# multiple samples
nt = rand(rng, d, 2)
@test nt.d isa Array{Float32,1}
@test size(nt.d) == (2,)
ℓ = logdensityof(d, nt)
@test ℓ isa Array{Float32,1}
@test size(ℓ) == (2,) 
