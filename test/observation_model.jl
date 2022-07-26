# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using CUDA
using MCMCDepth
using Plots
using Random

const PLOT = false
if PLOT
    plotly()
end
maybe_plot(fn, x...; y...) = PLOT ? fn(x...; y...) : nothing
rng = Random.default_rng()
Random.seed!(rng, 42)
curng = CUDA.default_rng()
Random.seed!(curng, 42)

# TODO move distribution generator function to main script / specific experiment script. Best practice: one script per experiment?
"""
    pixel_normal_exponential(min, max, σ, θ, μ, o)
Generate a Pixel distribution from the given parameters.
Putting static parameters first allows partial application of the function.
"""
function pixel_normal_exponential(min, max, σ, θ, μ, o)
    dist = KernelBinaryMixture(KernelNormal(μ, σ), KernelExponential(θ), o, one(o) - o)
    PixelDistribution(min, max, dist)
end
# WARN This might defeat chosen float precision
pixel_normal_exponential_default = pixel_normal_exponential | (0.1f0, 3.0f0, 0.01f0, 1.0f0)

# TODO Bottom up: Pixel, ImageModel, ObservationModel
# TODO Benchmark transfer time vs. inference time → double buffering worth it? If transfer is significant compared to inference?
pix_dist = pixel_normal_exponential_default(1.0f0, 0.1f0)
x = rand(CUDA.default_rng(), pix_dist, 100, 100)
maybe_plot(histogram, x |> flatten)
maybe_plot(plot, Gray.(x))
logdensityof(pix_dist, x) == logdensityof.(pix_dist, x)

# ImageModel
μ = rand(CUDA.default_rng(), KernelNormal(1.0f0, 0.01f0), 100, 100, 5)
maybe_plot(plot_depth_img, μ[:, :, 1])
o = rand(CUDA.default_rng(), KernelUniform(0.0f0, 0.5f0), 100, 100)
maybe_plot(plot_prob_img, o)
img_model = ImageModel(pixel_normal_exponential_default, μ, o, true)
imgs = rand(CUDA.default_rng(), img_model)
maybe_plot(plot_depth_img, imgs[:, :, 1])

# ObservationModel
using Accessors
using MCMCDepth
using CUDA
using LinearAlgebra
using SciGL
using Test
params = MCMCDepth.Parameters()
# TODO change to CuArray
render_context = RenderContext(params.width, params.height, params.depth, CuArray)
scene = Scene(params, render_context)

# CvCamera like ROS looks down positive z
t = [0, 0, 1.0]
r = normalize!([0.4, 0.3, 0.3, 0.3])
# TODO export
p = to_pose(t, r, QuatRotation)
μ = render(render_context, scene, 1, p)
@test size(μ) == (100, 100, 1)
@test eltype(μ) == Float32
@test maximum(μ) == 1.1870024f0

t_dist = [KernelNormal(0, 0.01), KernelNormal(0, 0.01), KernelNormal(1, 0.01)]
r_dist = [KernelNormal(), KernelNormal(), KernelNormal(), KernelNormal()]
T = rand(rng, t_dist, 10)
# QuatRotation takes care of normalization
R = rand(rng, r_dist, 10)
P = to_pose(T, R, QuatRotation)
μ_10 = render(render_context, scene, 1, P)
@test size(μ_10) == (100, 100, 10)
@test eltype(μ_10) == Float32
for slice in eachslice(μ_10, dims=2)
    @test 0 < maximum(slice) < 2
end
