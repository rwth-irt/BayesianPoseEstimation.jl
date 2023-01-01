# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 


include("../src/MCMCDepth.jl")
using .MCMCDepth

using Bijectors
using CUDA
using LinearAlgebra
using MCMCDepth
using Plots
using Quaternions
using Random
using Rotations
using StaticArrays
using Test

const PLOT = true
const N_SAMPLES = 50_000

# Yup 42 is bad style
curng = CUDA.RNG(42)
CUDA.allowscalar(false)
rng = Random.default_rng(42)


# Plot uniformity of rotation
if PLOT
    plotly()
    Q = @inferred rand(rng, dist, N_SAMPLES)
    Q_rot = to_rotation(Q)
    display(sphere_scatter(Q_rot))
    display(sphere_density((Q_rot)))
    # MCMCDepth.angleaxis_scatter((Q_rot)) |> display
end

# Compare with uniformly sampled Euler angles
if PLOT
    plotly()
    circ_dist = BroadcastedDistribution(_ -> KernelCircularUniform(), Vector{Float32}(undef, 3))
    Q = @inferred rand(rng, circ_dist, N_SAMPLES)
    Q_rot = to_rotation(Q)
    display(sphere_scatter(Q_rot, [1, 0, 0]))
    display(sphere_density(Q_rot, [1, 0, 0]))
    # MCMCDepth.angleaxis_scatter(Q_rot) |> display
end

# Additive Quaternion proposal
a = BroadcastedNode(:a, rng, QuaternionUniform, Float32)
b = DeterministicNode(:b, x -> 2 * x, (; a=a))
c = DeterministicNode(:c, x -> x, (; b=b))
prop = quaternion_additive(BroadcastedNode(:a, rng, QuaternionPerturbation, 0.01), c)
s1 = Sample(rand(b), MCMCDepth.quat_logp)
s2 = @inferred propose(prop, s1)
@test variables(s2).a != variables(s1).a
@test variables(s2).b == 2 * variables(s2).a
ℓ = @inferred transition_probability(prop, s1, s2)
@test ℓ != 0
@test ℓ == transition_probability(prop, s2, s1)

# multiple
S2 = @inferred propose(prop, s1, 2, 2)
@test variables(S2).a isa Matrix{QuaternionF64}
@test variables(S2).b == 2 * variables(S2).a
L = @inferred transition_probability(prop, s1, S2)
@test L != 0
@test L == transition_probability(prop, S2, s1)

# Symmetric Quaternion proposal
prop = quaternion_symmetric(BroadcastedNode(:a, rng, QuaternionPerturbation, 0.01), c)
s1 = Sample(rand(b), MCMCDepth.quat_logp)
s2 = @inferred propose(prop, s1)
@test variables(s2).a != variables(s1).a
@test variables(s2).b == 2 * variables(s2).a
ℓ = @inferred transition_probability(prop, s1, s2)
@test ℓ == 0
@test ℓ == transition_probability(prop, s2, s1)

# multiple
S2 = @inferred propose(prop, s1, 2, 2)
@test variables(S2).a isa Matrix{QuaternionF64}
@test variables(S2).b == 2 * variables(S2).a
L = @inferred transition_probability(prop, s1, S2)
@test L == 0
@test L == transition_probability(prop, S2, s1)
