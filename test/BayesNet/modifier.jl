# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# WARN Do not run this if you want Revise to work
include("../../src/MCMCDepth.jl")
using .MCMCDepth

using DensityInterface
using MCMCDepth
using Plots
using Random
using Test

# Minimal implementation to test whether the values get modified and the rest of the graph is traversed
struct SimpleModifierModel end
# Construct with same args as wrapped model
SimpleModifierModel(args...) = SimpleModifierModel()
Base.rand(::AbstractRNG, model::SimpleModifierModel, value) = 10 * value
DensityInterface.logdensityof(::SimpleModifierModel, ::Any, ℓ) = ℓ + one(ℓ)

plotly()
rng = Random.default_rng()

a = SimpleNode(:a, rng, KernelUniform())
b = SimpleNode(:b, rng, KernelExponential())
c = SimpleNode(:c, (; a=a, b=b), rng, KernelNormal)
d = SimpleNode(:d, (; c=c, b=b), rng, KernelNormal)
# TODO test if not at end
d_mod = ModifierNode(d, rng, SimpleModifierModel)

nt = rand(d_mod)
@test logdensityof(d, nt) == logdensityof(d_mod, nt) - 1
bij = bijector(d_mod)
@test bij isa NamedTuple{(:a, :b, :c, :d)}
@test values(bij) == (bijector(KernelUniform()), bijector(KernelExponential()), bijector(KernelNormal()), bijector(KernelNormal()))

# Visual test: d_mod should be wider than d
nt = rand(d_mod)
histogram([rand(d_mod).d for _ in 1:100]; label="d_mod");
histogram!([rand(d).d for _ in 1:100]; label="d")
