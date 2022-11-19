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

a = SimpleNode(:a, rng, KernelUniform())
b = SimpleNode(:b, rng, KernelExponential())
c = SimpleNode(:c, (; a=a, b=b), rng, KernelNormal)
d = SimpleNode(:d, (; c=c, b=b), rng, KernelNormal)

# Type stable rand
seq_graph = sequentialize(d)
nt = @inferred rand(seq_graph, (; a=1))
@test nt.a == 1
nt = @inferred rand(seq_graph)
@test nt.a != 1
ℓ = @inferred logdensityof(seq_graph, nt)
@test ℓ == logdensityof(KernelUniform(), nt.a) + logdensityof(KernelExponential(), nt.b) + logdensityof(KernelNormal(nt.a, nt.b), nt.c) + logdensityof(KernelNormal(nt.c, nt.b), nt.d)

# Test BroadcastedNode
a = BroadcastedNode(:a, rng, KernelUniform, 0, fill(1.0f0, 2))
b = BroadcastedNode(:b, rng, KernelExponential, fill(1.0f0, 2))
c = BroadcastedNode(:c, (; a=a, b=b), rng, KernelNormal)
d = BroadcastedNode(:d, (; c=c, b=b), rng, KernelNormal)

seq_graph = sequentialize(d)
nt = @inferred rand(seq_graph)
ℓ = @inferred logdensityof(seq_graph, nt)
@test ℓ ≈ sum(logdensityof.(KernelUniform(), nt.a) + logdensityof.(KernelExponential(), nt.b) + logdensityof.(KernelNormal.(nt.a, nt.b), nt.c) + logdensityof.(KernelNormal.(nt.c, nt.b), nt.d))

nt = @inferred rand(seq_graph, 3)
ℓ = @inferred logdensityof(seq_graph, nt)
@test ℓ' ≈ sum(logdensityof.(KernelUniform(), nt.a) + logdensityof.(KernelExponential(), nt.b) + logdensityof.(KernelNormal.(nt.a, nt.b), nt.c) + logdensityof.(KernelNormal.(nt.c, nt.b), nt.d); dims=(1,))

# evaluate deterministic nodes
fn(x, ::Any) = x
c = DeterministicNode(:c, fn, (; a=a, b=b))
d = BroadcastedNode(:d, (; c=c, b=b), rng, KernelNormal)
seq_graph = sequentialize(d)
@inferred evaluate(seq_graph, (; a=1, b=2.0f0, c=3.0f0, d=4.0f0))

# WARN non-simplified implementations are not type stable
@benchmark rand(d)
# 3-30x faster
@benchmark rand(seq_graph)

@benchmark rand(d, 100)
# 2x faster - gets less for larger dims
@benchmark rand(seq_graph, 100)

# For now no automatic broadcasting of logdensityof
vars = rand(seq_graph)
@benchmark logdensityof(d, vars)
# 3-30x faster
@benchmark logdensityof(seq_graph, vars)
