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

# TODO At what points in the graph does it make sense to reduce? Worst case is a dimension mismatch when accumulating the likelihoods
# 1. Leafs, e.g. translation / rotation
# 2. After deterministic functions which increase the dimensionality, e.g. rendering or NeuralNet decoders
# 3. After other broadcasted nodes
# TODO should these assumptions be coded, e.g. limit the constructor to the special cases?

a = MCMCDepth.ProductNode(:a, KernelUniform, fill(0.0, 3, 2), 1)
a_nt = @inferred rand(rng, a)
@test a_nt.a isa Array{Float64,2}
@test size(a_nt.a) == (3, 2)
a_ℓ = @inferred logdensityof(a, a_nt)
@test a_ℓ isa Float64

a_nt = @inferred rand(rng, a, 4)
@test a_nt.a isa Array{Float64,3}
@test size(a_nt.a) == (3, 2, 4)
a_ℓ = @inferred logdensityof(a, a_nt)
@test a_ℓ isa Array{Float64,1}
@test size(a_ℓ) == (4,)

# TODO Should leafs always be ProductNodes?
# b = MCMCDepth.BroadcastedNode(:b, KernelExponential, (2,), fill(1.0f0, 3, 2))
b = MCMCDepth.ProductNode(:b, KernelExponential, fill(1.0f0, 3, 2))
b_nt = @inferred rand(rng, b)
@test b_nt.b isa Array{Float32,2}
@test size(b_nt.b) == (3, 2)
b_ℓ = @inferred logdensityof(b, b_nt)
@test b_ℓ isa Float32

b_nt = @inferred rand(rng, b, 4)
@test b_nt.b isa Array{Float32,3}
@test size(b_nt.b) == (3, 2, 4)
b_ℓ = @inferred logdensityof(b, b_nt)
@test b_ℓ isa Array{Float32,1}
@test size(b_ℓ) == (4,)

# WARN sizes and reduction dims of children must be the same to combine them
# b = MCMCDepth.BroadcastedNode(:b, KernelExponential, (1, 2), fill(1.0f0, 3, 2))
c = VariableNode(:c, KernelNormal, (; a=a, b=b))
c_nt = rand(rng, c)
@test c_nt.c isa Array{Float64,2}
@test size(c_nt.c) == (3, 2)
c_ℓ = logdensityof(c, c_nt)
@test c_ℓ isa Array{Float64,2}
@test size(c_ℓ) == (3, 2)

# TODO this does not work... why should it?
# c_nt = rand(rng, c, 4)
# c_ℓ = logdensityof(c, c_nt)
# TODO Getting all the different combinations tested and thinking about which semantically make sense seems way too much work. Is it reasonable to focus on product leafs and matching roots?

# TODO why does it work and should it always work?
c_broad = BroadcastedNode(c, (2,))
cb_nt = rand(rng, c_broad)
@test cb_nt.c isa Array{Float64,2}
@test size(cb_nt.c) == (3, 2)
cb_ℓ = logdensityof(c_broad, cb_nt)
@test cb_ℓ isa Array{Float64,1}
# TODO why tho?
@test size(c_ℓ) == (3,2)

c_broad = BroadcastedNode(c, (1, 2))
cb_nt = rand(rng, c_broad)
@test cb_nt.c isa Array{Float64,2}
@test size(cb_nt.c) == (3, 2)
cb_ℓ = logdensityof(c_broad, cb_nt)
@test cb_ℓ isa Float64

cb_nt = rand(rng, c_broad, 4)
@test cb_nt.c isa Array{Float64,3}
@test size(cb_nt.c) == (3, 2, 4)
cb_ℓ = logdensityof(c_broad, cb_nt)
@test cb_ℓ isa Array{Float64,1}
@test size(cb_ℓ) == (4,)

# TEST NOT done yet
@test ℓ == logdensityof(KernelUniform(), nt.a) + logdensityof(KernelExponential(), nt.b) + logdensityof(KernelNormal(nt.a, nt.b), nt.c) + logdensityof(KernelNormal(nt.c, nt.b), nt.d)
bij = bijector(c_broad)
@test bij isa NamedTuple{(:a, :b, :c)}
@test values(bij) == (bijector(KernelUniform()), bijector(KernelExponential()), bijector(KernelNormal()), bijector(KernelNormal()))
