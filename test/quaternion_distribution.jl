# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 


include("../src/MCMCDepth.jl")
using .MCMCDepth

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

dist = QuaternionDistribution(Float64)

# Single
q = @inferred rand(rng, dist)
@test q isa Quaternion{Float64}
@test norm(q) ≈ 1
ℓ = @inferred logdensityof(dist, q)
@test ℓ isa Float64
@test ℓ == log(1 / π^2)

# Multiple
Q = @inferred rand(rng, dist, 3, 2)
@test Q isa Array{Quaternion{Float64},2}
@test size(Q) == (3, 2)
@test reduce(&, norm.(Q) .≈ 1)
ℓ = @inferred logdensityof(dist, Q)
@test ℓ isa Array{Float64}
@test size(ℓ) == (3, 2)
@test reduce(&, ℓ .== log(1 / π^2))

# CUDA
Q = @inferred rand(curng, dist, 100)
@test Q isa CuArray{Quaternion{Float64},1}
ℓ = @inferred logdensityof(dist, Q)
@test ℓ isa CuVector{Float64}
@test reduce(&, ℓ .== log(1 / π^2))

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
    circ_dist = ProductBroadcastedDistribution(_ -> KernelCircularUniform(), Vector{Float32}(undef, 3))
    Q = @inferred rand(rng, circ_dist, N_SAMPLES)
    Q_rot = to_rotation(Q)
    display(sphere_scatter(Q_rot, [1, 0, 0]))
    display(sphere_density(Q_rot, [1, 0, 0]))
    # MCMCDepth.angleaxis_scatter(Q_rot) |> display
end

# Quaternion perturbation
q0 = Quaternion(Float16(1), 0, 0, 0, true)
pert = QuaternionPerturbation(0.1f0)
q1 = @inferred rand(rng, pert)
ℓ = @inferred logdensityof(pert, q1)
@test ℓ == sum(logdensityof.(KernelNormal(0, 0.1f0), imag_part(q1) .* 2))
@test norm(q1) ≈ 1
@test q0 * q1 isa QuaternionF32
@test q1 * q0 == q1
q2 = rand(rng, dist)
@test q1 * q2 isa QuaternionF64
@test q1 * q2 != q2 * q1
# multiple
Q = @inferred rand(rng, pert, 2, 2)
@test Q isa Matrix{QuaternionF32}
L = @inferred logdensityof(pert, Q)
@test L isa Matrix{Float32}

# Normalization approximation
ϕ = rand(KernelNormal(0, 0.01f0), 3)
@test isapprox(qrotation(ϕ), MCMCDepth.approx_qrotation(ϕ...))

# QuaternionProposal
a = BroadcastedNode(:a, rng, QuaternionDistribution, Float32)
b = DeterministicNode(:b, x -> 2 * x, (; a=a))
c = DeterministicNode(:c, x -> x, (; b=b))
rand(b)
prop = QuaternionProposal(BroadcastedNode(:a, rng, QuaternionPerturbation, 0.01), c)
s1 = Sample(rand(b), MCMCDepth.quat_logp)
s2 = @inferred propose(prop, s1)
@test variables(s2).a != variables(s1).a
@test variables(s2).b == 2 * variables(s2).a
ℓ = @inferred transition_probability(prop, s1, s2)
@test ℓ == 0

# multiple
S2 = @inferred propose(prop, s1, 2, 2)
@test variables(S2).a isa Matrix{QuaternionF64}
@test variables(S2).b == 2 * variables(S2).a
L = @inferred transition_probability(prop, s1, S2)
@test L == 0
