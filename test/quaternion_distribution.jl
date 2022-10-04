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

const PLOT = false

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
    Q = @inferred rand(rng, dist, 50_000)
    aa_rot = AngleAxis.(to_rotation(Q, QuatRotation))
    x = getproperty.(aa_rot, (:axis_x,))
    y = getproperty.(aa_rot, (:axis_y,))
    z = getproperty.(aa_rot, (:axis_z,))
    theta = getproperty.(aa_rot, (:theta,))
    scatter3d(x, y, z; color=:blue, markersize=0.1)
end

if PLOT
    pyplot()
    vecs = aa_rot .* ([0, 1, 0],)
    z_angles = atan.(getindex.(vecs, 1), getindex.(vecs, 2))
    histogram(z_angles; proj=:polar, fill=true, nbins=90)
end

# Compare with uniformly sampled Euler angles
circ_dist = ProductBroadcastedDistribution(_ -> KernelCircularUniform(), Vector{Float32}(undef, 3))
if PLOT
    plotly()
    Q = @inferred rand(rng, circ_dist, 50_000)
    aa_rot = AngleAxis.(to_rotation(Q, RotXYZ))
    x = getproperty.(aa_rot, (:axis_x,))
    y = getproperty.(aa_rot, (:axis_y,))
    z = getproperty.(aa_rot, (:axis_z,))
    theta = getproperty.(aa_rot, (:theta,))
    scatter3d(x, y, z; color=:blue, markersize=0.1)
end

if PLOT
    pyplot()
    vecs = aa_rot .* ([0, 1, 0],)
    z_angles = atan.(getindex.(vecs, 1), getindex.(vecs, 2))
    histogram(z_angles; proj=:polar, fill=true, nbins=90)
end


R = @inferred to_rotation(rand(rng, circ_dist, 30_000), RotXYZ{Float32})
# p = SVector{3,}(0, 0, 1.0f0)
p = SVector{3,}(1.0f0, 0, 0)
R_p = map(r -> Vector(r * p), R)

plotlyjs()
R_plot = hcat(R_p...)
scatter3d(R_plot[1, :], R_plot[2, :], R_plot[3, :]; markersize=1.0, markeralpha=0.3)

using CoordinateTransformations
plotlyjs()
sfc = SphericalFromCartesian()
sph = sfc.(R_p)
# histogram2d(getproperty.(sph, :θ), getproperty.(sph, :ϕ); nbins=(45, 45))

using StatsBase
n = 45
# his = fit(Histogram, getproperty.(sph, :θ), (range(-π, π; length=n), range(-π, π; length=n + 1)))
# TODO smoother with KDE?
his = fit(Histogram, (getproperty.(sph, :θ), getproperty.(sph, :ϕ)), (range(-π, π; length=n + 1), range(-π / 2, π / 2; length=n + 1)))
# heatmap(his.weights')

# Plot histogram on sphere
using PlotlyJS
u = range(-π, π; length=n)
v = range(0, π; length=n)
x = cos.(u) * sin.(v)'
y = sin.(u) * sin.(v)'
z = ones(n) * cos.(v)'
PlotlyJS.surface(x=x, y=y, z=z, surfacecolor=his.weights) |> PlotlyJS.plot
