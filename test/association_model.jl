# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using CUDA
using Distributions
using LinearAlgebra
using MCMCDepth
using Plots
using Random
using SciGL
using Test

const PLOT = true
if PLOT
    pyplot()
end
maybe_plot(fn, x...; y...) = PLOT ? fn(x...; y...) : nothing
rng = Random.default_rng()
Random.seed!(rng, 42)
curng = CUDA.default_rng()
Random.seed!(curng, 42)
CUDA.allowscalar(false)

# Parameters
params = MCMCDepth.Parameters()
render_context = RenderContext(params.width, params.height, params.depth, CuArray)
scene = Scene(params, render_context)
t = [0, 0, 1.5]
r = normalize!([1, 0, 0, 0])
p = to_pose(t, r, QuatRotation)

# WARN 0.01 quickly leads to 0 probability
const σ = 0.1f0
const θ = 1.0f0
dist_is(σ, μ) = PixelDistribution(μ, KernelNormal(μ, σ))
my_dist_is = dist_is | (σ)
# TEST a truncated dist_not should really help to estimate the association
dist_not(θ, μ) = PixelDistribution(μ, truncated(KernelExponential(θ), nothing, μ))
my_dist_not = dist_not | (θ)

# PixelAssociation
μ = 1.0f0
prior = 0.3f0
pa = PixelAssociation(prior, my_dist_is(μ), my_dist_not(μ))

samples = @inferred rand(rng, pa, 10000)
@test samples isa Vector{Float32}
@test size(samples) == (10000,)
maybe_plot(histogram, samples)

ℓ = @inferred logdensityof(pa, samples)
@test ℓ isa Vector{Float32}
@test size(ℓ) == size(samples)
@test minimum(ℓ) >= 0.0
@test maximum(ℓ) <= 1.0
@test 0.0 <= mean(ℓ) <= 1.0

# Mean should be the association probability which generated the samples
@test isapprox(mean(ℓ), prior; atol=0.01)
prior = 0.8f0
pa = PixelAssociation(prior, my_dist_is(μ), my_dist_not(μ))
values = @inferred rand(rng, pa, 10000)
@test isapprox(mean(logdensityof(pa, values)), prior; atol=0.01)

# Invalid depth values should return prior
@test logdensityof(pa, -eps(μ)) == prior
@test logdensityof(pa, 0) == prior
@test logdensityof(pa, eps(Float32)) != prior
# Values out of support of dist_not should return 1
@test logdensityof(pa, μ) != 1
@test logdensityof(pa, μ + eps(μ)) == 1

# ImageAssociation
Q_o = CUDA.fill(0.5f0, 100, 100)
μ = render(render_context, scene, 1, p)
maybe_plot(plot_depth_img, Array(μ))
ia = ImageAssociation(my_dist_is, my_dist_not, Q_o, μ)

img = @inferred rand(curng, ia)
@test img isa CuMatrix{Float32}
@test size(img) == (100, 100)
maybe_plot(plot_depth_img, Array(img))

# TODO this would be expected for a scalar o → extract Dims from given prior
# TODO test scalar o in ObservationModel
ℓ = @inferred logdensityof(ia, img)
@test ℓ isa CuMatrix{Float32}
@test size(ℓ) == size(img)
@test minimum(ℓ) >= 0.0
@test maximum(ℓ) <= 1.0
@test 0.0 <= mean(ℓ) <= 1.0
@test isapprox(mean(ℓ), mean(Q_o); atol=0.01)
maybe_plot(plot_prob_img, Array(ℓ))
