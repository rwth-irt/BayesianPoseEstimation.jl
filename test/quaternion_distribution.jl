# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 


# TODO
include("../src/MCMCDepth.jl")
using .MCMCDepth

using CUDA
using LinearAlgebra
using MCMCDepth
using Rotations
using Plots
using Random
using Test

const PLOT = true

# Yup 42 is bad style
curng = CUDA.RNG(42)
CUDA.allowscalar(false)
rng = Random.default_rng(42)

dist = QuaternionDistribution(Float64)

# Single
q = @inferred rand(rng, dist)
@test q isa Vector{Float64}
@test norm(q) ≈ 1
ℓ = @inferred logdensityof(dist, q)
@test ℓ isa Float64
@test ℓ == log(1 / π^2)

# Multiple
Q = @inferred rand(rng, dist, 3, 2)
@test Q isa Array{Float64}
@test size(Q) == (4, 3, 2)
@test reduce(&, norm_dims(Q) .≈ 1)
ℓ = @inferred logdensityof(dist, Q)
@test ℓ isa Array{Float64}
@test size(ℓ) == (3, 2)
@test reduce(&, ℓ .== log(1 / π^2))

# CUDA
Q = @inferred rand(curng, dist, 100)
@test Q isa CuMatrix{Float64}
ℓ = @inferred logdensityof(dist, Q)
@test ℓ isa CuVector{Float64}
@test reduce(&, ℓ .== log(1 / π^2))

# Plot uniformity of rotation
if PLOT
    plotly()
    Q = @inferred rand(rng, dist, 5000)
    aa_rot = AngleAxis.(to_rotation(Q, QuatRotation))
    x = getproperty.(aa_rot, (:axis_x,))
    y = getproperty.(aa_rot, (:axis_y,))
    z = getproperty.(aa_rot, (:axis_z,))
    theta = getproperty.(aa_rot, (:theta,))
    scatter3d(x, y, z; color=:blue, markersize=1)
end

if PLOT
    pyplot()
    vecs = aa_rot .* ([0, 1, 0],)
    z_angles = atan.(getindex.(vecs, 1), getindex.(vecs, 2))
    histogram(z_angles; proj=:polar, fill=true, nbins=90)
end
