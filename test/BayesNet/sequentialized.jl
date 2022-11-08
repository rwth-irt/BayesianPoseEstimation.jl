# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# WARN Do not run this if you want Revise to work
include("../../src/MCMCDepth.jl")
using .MCMCDepth

using BenchmarkTools
using MCMCDepth
using Random
using Test

rng = Random.default_rng()

a = SimpleNode(:a, KernelUniform())
b = SimpleNode(:b, KernelExponential())
c = SimpleNode(:c, KernelNormal, (; a=a, b=b))
d = SimpleNode(:d, KernelNormal, (; c=c, b=b))

seq_graph = sequentialize(d)
nt = @inferred rand(Random.default_rng(), seq_graph)
ℓ = @inferred logdensityof(seq_graph, nt)
@test ℓ == logdensityof(KernelUniform(), nt.a) + logdensityof(KernelExponential(), nt.b) + logdensityof(KernelNormal(nt.a, nt.b), nt.c) + logdensityof(KernelNormal(nt.c, nt.b), nt.d)

# Test BroadcastedNode
a = BroadcastedNode(:a, KernelUniform, 0, fill(1.0f0, 2))
b = BroadcastedNode(:b, KernelExponential, fill(1.0f0, 2))
c = BroadcastedNode(:c, KernelNormal, (; a=a, b=b))
d = BroadcastedNode(:d, KernelNormal, (; c=c, b=b))

seq_graph = sequentialize(d)
nt = @inferred rand(Random.default_rng(), seq_graph)
ℓ = @inferred logdensityof(seq_graph, nt)
@test ℓ == sum(logdensityof.(KernelUniform(), nt.a) + logdensityof.(KernelExponential(), nt.b) + logdensityof.(KernelNormal.(nt.a, nt.b), nt.c) + logdensityof.(KernelNormal.(nt.c, nt.b), nt.d))

nt = @inferred rand(Random.default_rng(), seq_graph, 3)
ℓ = @inferred logdensityof(seq_graph, nt)
@test ℓ' ≈ sum(logdensityof.(KernelUniform(), nt.a) + logdensityof.(KernelExponential(), nt.b) + logdensityof.(KernelNormal.(nt.a, nt.b), nt.c) + logdensityof.(KernelNormal.(nt.c, nt.b), nt.d); dims=(1,))

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
