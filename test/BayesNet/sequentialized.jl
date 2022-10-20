# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using BenchmarkTools
using MCMCDepth
using Random
using Test

rng = Random.default_rng()

a = VariableNode(:a, KernelUniform, (;))
b = VariableNode(:b, KernelExponential, (;))
c = VariableNode(:c, KernelNormal, (; a=a, b=b))
d = VariableNode(:d, KernelNormal, (; c=c, b=b))

seq_graph = sequentialize(d)
rand(Random.default_rng(), seq_graph)
nt = @inferred rand(Random.default_rng(), seq_graph)
ℓ = @inferred logdensityof(seq_graph, nt)
@test ℓ == logdensityof(KernelUniform(), nt.a) + logdensityof(KernelExponential(), nt.b) + logdensityof(KernelNormal(nt.a, nt.b), nt.c) + logdensityof(KernelNormal(nt.c, nt.b), nt.d)

# WARN non-simplified implementations are not type stable
@benchmark rand(Random.default_rng(), d)
# 30x faster
@benchmark rand(Random.default_rng(), seq_graph)

@benchmark rand(Random.default_rng(), d, 100)
# 2x faster - gets less for larger dims
@benchmark rand(Random.default_rng(), seq_graph, 100)

# For now no automatic broadcasting of logdensityof
vars = rand(Random.default_rng(), seq_graph)
@benchmark logdensityof(d, vars)
# 30x faster
@benchmark logdensityof(seq_graph, vars)