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
const N_SAMPLES = 20_000

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
    Q_rot = to_rotation(Q, QuatRotation)
    sphere_scatter(Q_rot) |> display
    sphere_density((Q_rot)) |> display
end

# Compare with uniformly sampled Euler angles
if PLOT
    plotly()
    circ_dist = ProductBroadcastedDistribution(_ -> KernelCircularUniform(), Vector{Float32}(undef, 3))
    Q = @inferred rand(rng, circ_dist, N_SAMPLES)
    Q_rot = to_rotation(Q)
    sphere_scatter(Q_rot) |> display
    sphere_density(Q_rot) |> display
end
