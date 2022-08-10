# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using Accessors
using CUDA
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

# TODO move distribution generator function to main script / specific experiment script. Best practice: one script per experiment?
"""
    pixel_normal_exponential(min, max, σ, θ, μ, o)
Generate a Pixel distribution from the given parameters.
Putting static parameters first allows partial application of the function.
"""
function pixel_normal_exponential(min, max, σ, θ, μ, o)
    # TODO Generating random "occlusions" behind the object does not make sense. Would need some kind of truncated exponential. However, it might be more robust, since it allows for random outliers.
    dist = KernelBinaryMixture(KernelNormal(μ, σ), KernelExponential(θ), o, one(o) - o)
    PixelDistribution(min, max, dist)
end
# WARN This might defeat chosen float precision
pixel_normal_exponential_default = pixel_normal_exponential | (0.1f0, 3.0f0, 0.01f0, 1.0f0)

# TODO Benchmark transfer time vs. inference time → double buffering worth it? If transfer is significant compared to inference?
pix_dist = pixel_normal_exponential_default(1.0f0, 0.1f0)
x = rand(curng, pix_dist, 100, 100)
maybe_plot(histogram, x |> Array |> flatten)
maybe_plot(plot_depth_img, Array(x))
logdensityof(pix_dist, x) == logdensityof.(pix_dist, x)

# ImageModel

μ = rand(curng, KernelNormal(1.0f0, 0.01f0), 100, 100, 5)
maybe_plot(plot_depth_img, Array(@view μ[:, :, 1]))
# Test overriding clims
o = rand(curng, KernelUniform(0.5f0, 1.0f0), 100, 100)
maybe_plot(plot_prob_img, Array(o), clims=nothing)
# Test normal scaling
maybe_plot(plot_prob_img, Array(o))
# Higher association probability so the ape can be recognized
o = rand(curng, KernelUniform(0.0f0, 1.0f0), 100, 100)
img_model = ImageModel(pixel_normal_exponential_default, μ, o, true)
imgs = rand(curng, img_model)
@test size(imgs) == (100, 100, 5)
ℓ = @inferred logdensityof(img_model, imgs)
@test size(ℓ) == (5,)
@test eltype(ℓ) == Float32
maybe_plot(plot_depth_img, Array(@view imgs[:, :, 1]))

# RenderContext with ImageModel

params = MCMCDepth.Parameters()
render_context = RenderContext(params.width, params.height, params.depth, CuArray)
scene = Scene(params, render_context)

# CvCamera like ROS looks down positive z
t = [0, 0, 1.5]
r = normalize!([1, 0, 0, 0])
p = to_pose(t, r, QuatRotation)
μ = render(render_context, scene, 1, p)
@test size(μ) == (100, 100)
@test eltype(μ) == Float32
@test maximum(μ) == 1.7077528f0
maybe_plot(plot_depth_img, Array(μ))

# Single rand
img_model = ImageModel(pixel_normal_exponential_default, μ, o, true)
img = rand(curng, img_model)
@test size(img) == (100, 100)
@test eltype(img) == Float32
maybe_plot(plot_depth_img, Array(img))
ℓ = @inferred logdensityof(img_model, img)
@test ℓ isa Float32

# Multiple rand image model
img_10 = rand(curng, img_model, 10)
@test size(img_10) == (100, 100, 10)
@test eltype(img_10) == Float32
maybe_plot(plot_depth_img, Array(view(img_10, :, :, 10)))
ℓ = @inferred logdensityof(img_model, img_10)
@test size(ℓ) == (10,)
@test ℓ isa CuArray{Float32}
for layer_id in 1:(size(img_10)[3]-1)
    @test @views img_10[:, :, layer_id] != img_10[:, :, layer_id+1]
    Array(ℓ)[layer_id] != Array(ℓ)[layer_id+1]
end

# Multiple random poses
t_dist = [KernelNormal(0, 0.01), KernelNormal(0, 0.01), KernelNormal(1.5, 0.01)]
r_dist = [KernelNormal(), KernelNormal(), KernelNormal(), KernelNormal()]
T = rand(rng, t_dist, 10)
# QuatRotation takes care of normalization
R = rand(rng, r_dist, 10)
P = to_pose(T, R, QuatRotation)
μ_10 = render(render_context, scene, 1, P)
@test size(μ_10) == (100, 100, 10)
@test eltype(μ_10) == Float32
for layer_id in 1:(size(μ_10)[3]-1)
    @test @views μ_10[:, :, layer_id] != μ_10[:, :, layer_id+1]
end

# Multiple poses & rand image model
img_model_10 = ImageModel(pixel_normal_exponential_default, μ_10, o, true)
img_10_2 = rand(curng, img_model_10, 2)
@test size(img_10_2) == (100, 100, 10, 2)
@test eltype(img_10_2) == Float32
maybe_plot(plot_depth_img, Array(view(img_10_2, :, :, 10, 2)))
ℓ = @inferred logdensityof(img_model_10, img_10_2)
@test size(ℓ) == (10, 2)
@test ℓ isa CuArray{Float32}
for layer_id in 1:(size(img_10_2)[3])
    for rand_id in 1:(size(img_10_2)[4]-1)
        @test @views img_10_2[:, :, layer_id, rand_id] != img_10_2[:, :, layer_id, rand_id+1]
        Array(ℓ)[layer_id, rand_id] != Array(ℓ)[layer_id, rand_id+1]
    end
end
for layer_id in 1:(size(img_10_2)[3]-1)
    for rand_id in 1:(size(img_10_2)[4])
        @test @views img_10_2[:, :, layer_id, rand_id] != img_10_2[:, :, layer_id+1, rand_id]
        Array(ℓ)[layer_id, rand_id] != Array(ℓ)[layer_id+1, rand_id]
    end
end

# ObservationModel

# WARN lots of things have to be right when using eval of params
params = @set params.rotation_type = QuatRotation
params = @set params.pixel_dist = pixel_normal_exponential_default
obs_model = ObservationModel(params, render_context, scene, t, r, o)
μ = render(obs_model)
@test size(μ) == (100, 100)
@test eltype(μ) == Float32
maybe_plot(plot_depth_img, Array(μ))

# Single random noise
img = rand(curng, obs_model)
@test size(img) == (100, 100)
@test eltype(img) == Float32
maybe_plot(plot_depth_img, Array(img))
ℓ = @inferred logdensityof(obs_model, img)
@test ℓ isa Float32

# Multiple random noise
img_10 = rand(curng, obs_model, 10)
@test size(img_10) == (100, 100, 10)
@test eltype(img_10) == Float32
ℓ = @inferred logdensityof(obs_model, img_10)
@test size(ℓ) == (10,)
@test ℓ isa CuArray{Float32}
for layer_id in 1:(size(img_10)[3]-1)
    @test @views img_10[:, :, layer_id] != img_10[:, :, layer_id+1]
    @test Array(ℓ)[layer_id] != Array(ℓ)[layer_id+1]
end

# Multiple Poses
t_dist = BroadcastedDistribution(KernelNormal, [0, 0, 1.5], [0.01, 0.01, 0.01])
r_dist = BroadcastedDistribution(KernelNormal, [0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0])
T = rand(t_dist, 10)
R = rand(r_dist, 10)

obs_model = ObservationModel(params, render_context, scene, T, R, o)
μ = render(obs_model)
@test size(μ) == (100, 100, 10)
@test eltype(μ) == Float32
maybe_plot(plot_depth_img, Array(view(μ, :, :, 4)))

img_10 = rand(curng, obs_model)
@test size(img_10) == (100, 100, 10)
@test eltype(img_10) == Float32
ℓ = @inferred logdensityof(obs_model, img_10)
@test size(ℓ) == (10,)
@test ℓ isa CuArray{Float32}
for layer_id in 1:(size(img_10)[3]-1)
    @test @views img_10[:, :, layer_id] != img_10[:, :, layer_id+1]
    @test Array(ℓ)[layer_id] != Array(ℓ)[layer_id+1]
end
maybe_plot(plot_depth_img, Array(view(img_10, :, :, 4)))

img_10_2 = rand(curng, obs_model, 2)
@test size(img_10_2) == (100, 100, 10, 2)
@test eltype(img_10_2) == Float32
ℓ = @inferred logdensityof(obs_model, img_10_2)
@test size(ℓ) == (10, 2)
@test ℓ isa CuArray{Float32}
for layer_id in 1:(size(img_10_2)[3])
    for rand_id in 1:(size(img_10_2)[4]-1)
        @test @views img_10_2[:, :, layer_id, rand_id] != img_10_2[:, :, layer_id, rand_id+1]
        Array(ℓ)[layer_id, rand_id] != Array(ℓ)[layer_id, rand_id+1]
    end
end
for layer_id in 1:(size(img_10_2)[3]-1)
    for rand_id in 1:(size(img_10_2)[4])
        @test @views img_10_2[:, :, layer_id, rand_id] != img_10_2[:, :, layer_id+1, rand_id]
        @test Array(ℓ)[layer_id, rand_id] != Array(ℓ)[layer_id+1, rand_id]
    end
end

# Multiple associations
O = rand(curng, KernelUniform(0.0f0, 0.9f0), 100, 100, 10)
obs_model = ObservationModel(params, render_context, scene, t, r, O)
μ = render(obs_model)
@test size(μ) == (100, 100)
@test eltype(μ) == Float32
maybe_plot(plot_depth_img, Array(μ))

img_10 = rand(curng, obs_model)
@test size(img_10) == (100, 100, 10)
@test eltype(img_10) == Float32
ℓ = @inferred logdensityof(obs_model, img_10)
@test size(ℓ) == (10,)
@test ℓ isa CuArray{Float32}
for layer_id in 1:(size(img_10)[3]-1)
    @test @views img_10[:, :, layer_id] != img_10[:, :, layer_id+1]
    @test Array(ℓ)[layer_id] != Array(ℓ)[layer_id+1]
end
maybe_plot(plot_depth_img, Array(view(img_10, :, :, 5)))

img_10_2 = rand(curng, obs_model, 2)
@test size(img_10_2) == (100, 100, 10, 2)
@test eltype(img_10_2) == Float32
ℓ = @inferred logdensityof(obs_model, img_10_2)
@test size(ℓ) == (10, 2)
@test ℓ isa CuArray{Float32}
for layer_id in 1:(size(img_10_2)[3])
    for rand_id in 1:(size(img_10_2)[4]-1)
        @test @views img_10_2[:, :, layer_id, rand_id] != img_10_2[:, :, layer_id, rand_id+1]
        Array(ℓ)[layer_id, rand_id] != Array(ℓ)[layer_id, rand_id+1]
    end
end
for layer_id in 1:(size(img_10_2)[3]-1)
    for rand_id in 1:(size(img_10_2)[4])
        @test @views img_10_2[:, :, layer_id, rand_id] != img_10_2[:, :, layer_id+1, rand_id]
        @test Array(ℓ)[layer_id, rand_id] != Array(ℓ)[layer_id+1, rand_id]
    end
end

# Multiple poses & associations
O = rand(curng, KernelUniform(0.0f0, 0.9f0), 100, 100, 5)
obs_model = ObservationModel(params, render_context, scene, T, R, O)
@test_throws DimensionMismatch img_10 = rand(curng, obs_model)

O = rand(curng, KernelUniform(0.0f0, 0.9f0), 100, 100, 10)
obs_model = ObservationModel(params, render_context, scene, T, R, O)
img_10 = rand(curng, obs_model)
@test size(img_10) == (100, 100, 10)
@test eltype(img_10) == Float32
ℓ = @inferred logdensityof(obs_model, img_10)
@test size(ℓ) == (10,)
@test ℓ isa CuArray{Float32}
for layer_id in 1:(size(img_10)[3]-1)
    @test @views img_10[:, :, layer_id] != img_10[:, :, layer_id+1]
    @test Array(ℓ)[layer_id] != Array(ℓ)[layer_id+1]
end
maybe_plot(plot_depth_img, Array(view(img_10, :, :, 8)))

img_10_2 = rand(curng, obs_model, 2)
@test size(img_10_2) == (100, 100, 10, 2)
@test eltype(img_10_2) == Float32
ℓ = @inferred logdensityof(obs_model, img_10_2)
@test size(ℓ) == (10, 2)
@test ℓ isa CuArray{Float32}
for layer_id in 1:(size(img_10_2)[3])
    for rand_id in 1:(size(img_10_2)[4]-1)
        @test @views img_10_2[:, :, layer_id, rand_id] != img_10_2[:, :, layer_id, rand_id+1]
        Array(ℓ)[layer_id, rand_id] != Array(ℓ)[layer_id, rand_id+1]
    end
end
for layer_id in 1:(size(img_10_2)[3]-1)
    for rand_id in 1:(size(img_10_2)[4])
        @test @views img_10_2[:, :, layer_id, rand_id] != img_10_2[:, :, layer_id+1, rand_id]
        @test Array(ℓ)[layer_id, rand_id] != Array(ℓ)[layer_id+1, rand_id]
    end
end
