# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using Accessors
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

# Setup render context & scene
params = MCMCDepth.Parameters()
params = @set params.mesh_files = ["meshes/BM067R.obj"]
render_context = RenderContext(params.width, params.height, params.depth, CuArray)

# CvCamera like ROS looks down positive z
scene = Scene(params, render_context)
t = [-0.05, 0.05, 0.25]
r = normalize([1, 1, 0, 1])
p = to_pose(t, r, QuatRotation)
μ = render(render_context, scene, 1, p)
maybe_plot(plot_depth_img, Array(μ))
@test size(μ) == (100, 100)
@test eltype(μ) == Float32
@test maximum(μ) > 0

# TODO Benchmark transfer time vs. inference time → double buffering worth it? If transfer is significant compared to inference?

# PixelDistribution

"""
    pixel_normal_exponential(min, max, σ, θ, μ, o)
Generate a Pixel distribution from the given parameters.
Putting static parameters first allows partial application of the function.
"""
function mix_normal_truncated_exponential(σ::T, θ::T, μ::T, o::T) where {T<:Real}
    # TODO should these generators be part of experiment specific scripts or should I provide some default ones?
    # TODO Compare whether truncated even makes a difference
    dist = KernelBinaryMixture(KernelNormal(μ, σ), truncated(KernelExponential(θ), nothing, μ), o, one(o) - o)
    PixelDistribution(μ, dist)
end

my_pixel_dist = mix_normal_truncated_exponential | (0.5f0, 0.5f0)
# Exponential truncated to 0.0 is problematic, invalid values of μ should be ignored
pix_dist = my_pixel_dist(0.0f0, 0.1f0)
x = rand(curng, pix_dist, 100, 100)
@test maximum(x) == 0
@test minimum(x) == 0

@test logdensityof(pix_dist, 1.0) == 0
@test logdensityof(pix_dist, 0.0) == 0
@test logdensityof(pix_dist, -eps(Float32)) == 0

# Valid μ range
pix_dist = my_pixel_dist(0.5f0, 0.5f0)
x = rand(curng, pix_dist, 100, 100)
@test minimum(x) >= 0
maybe_plot(histogram, x |> Array |> flatten)

@test logdensityof(pix_dist, 1.0) != 0
@test logdensityof(pix_dist, 0.0) == 0
@test logdensityof(pix_dist, -eps(Float32)) == 0

ℓ = @inferred logdensityof(pix_dist, x)
@test minimum(ℓ) != 0
@test ℓ == logdensityof.(pix_dist, x)
@test !isinf(sum(ℓ))
maybe_plot(histogram, ℓ |> Array |> flatten)

# Single rand
o = rand(curng, KernelUniform(0.8f0, 1.0f0), 100, 100)
obs_model = ObservationModel(true, my_pixel_dist, μ, o)
img = rand(curng, obs_model)
@test size(img) == (100, 100)
@test eltype(img) == Float32
maybe_plot(plot_depth_img, Array(img))
ℓ = @inferred logdensityof(obs_model, img)
@test !isinf(ℓ)
@test ℓ isa Float32
maybe_plot(plot_depth_img, logdensityof.(my_pixel_dist.(μ, o), img) .|> exp |> Array; colorbar_title="logdensity", reverse=false, value_to_typemax=1)
# Shorter differences should result in higher logdensity. Keep in mind the mixture.
maybe_plot(plot_depth_img, img .- μ |> Array; colorbar_title="depth difference [m]")

# Multiple rand image model
img_10 = rand(curng, obs_model, 10)
@test size(img_10) == (100, 100, 10)
@test eltype(img_10) == Float32
maybe_plot(plot_depth_img, Array(view(img_10, :, :, 10)))
ℓ = @inferred logdensityof(obs_model, img_10)
@test size(ℓ) == (10,)
@test ℓ isa CuArray{Float32}
for layer_id in 1:(size(img_10)[3]-1)
    @test @views img_10[:, :, layer_id] != img_10[:, :, layer_id+1]
    Array(ℓ)[layer_id] != Array(ℓ)[layer_id+1]
end

# Multiple random poses
t_dist = [KernelNormal(t[1], 0.01), KernelNormal(t[2], 0.01), KernelNormal(t[3], 0.01)]
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
obs_model_10 = ObservationModel(true, my_pixel_dist, μ_10, o)
img_10_2 = rand(curng, obs_model_10, 2)
@test size(img_10_2) == (100, 100, 10, 2)
@test eltype(img_10_2) == Float32
maybe_plot(plot_depth_img, Array(view(img_10_2, :, :, 10, 2)))
ℓ = @inferred logdensityof(obs_model_10, img_10_2)
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

# ImageModel from scene & pose
# WARN lots of things have to be right when using eval of params
params = @set params.rotation_type = QuatRotation
params = @set params.pixel_dist = my_pixel_dist
obs_model_fn = ObservationModel | (params, render_context, scene)
obs_model_tro = obs_model_fn(t, r, o)
@test size(obs_model_tro.μ) == (100, 100)
@test eltype(obs_model_tro.μ) == Float32

# Single random noise
img = rand(curng, obs_model_tro)
@test size(img) == (100, 100)
@test eltype(img) == Float32
maybe_plot(plot_depth_img, Array(img); clims=(0.0, 2.0))
ℓ = @inferred logdensityof(obs_model_tro, img)
@test ℓ isa Float32

# Multiple random noise
img_10 = rand(curng, obs_model_tro, 10)
@test size(img_10) == (100, 100, 10)
@test eltype(img_10) == Float32
ℓ = @inferred logdensityof(obs_model_tro, img_10)
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

obs_model_TRo = obs_model_fn(T, R, o)
@test size(obs_model_TRo.μ) == (100, 100, 10)
@test eltype(obs_model_TRo.μ) == Float32

img_10 = rand(curng, obs_model_TRo)
@test size(img_10) == (100, 100, 10)
@test eltype(img_10) == Float32
ℓ = @inferred logdensityof(obs_model_TRo, img_10)
@test size(ℓ) == (10,)
@test ℓ isa CuArray{Float32}
for layer_id in 1:(size(img_10)[3]-1)
    @test @views img_10[:, :, layer_id] != img_10[:, :, layer_id+1]
    @test Array(ℓ)[layer_id] != Array(ℓ)[layer_id+1]
end
maybe_plot(plot_depth_img, Array(view(img_10, :, :, 4)))

img_10_2 = rand(curng, obs_model_TRo, 2)
@test size(img_10_2) == (100, 100, 10, 2)
@test eltype(img_10_2) == Float32
ℓ = @inferred logdensityof(obs_model_TRo, img_10_2)
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
obs_model_trO = obs_model_fn(t, r, O)
@test size(obs_model_trO.μ) == (100, 100)
@test eltype(obs_model_trO.μ) == Float32

img_10 = rand(curng, obs_model_trO)
@test size(img_10) == (100, 100, 10)
@test eltype(img_10) == Float32
ℓ = @inferred logdensityof(obs_model_trO, img_10)
@test size(ℓ) == (10,)
@test ℓ isa CuArray{Float32}
for layer_id in 1:(size(img_10)[3]-1)
    @test @views img_10[:, :, layer_id] != img_10[:, :, layer_id+1]
    @test Array(ℓ)[layer_id] != Array(ℓ)[layer_id+1]
end
maybe_plot(plot_depth_img, Array(view(img_10, :, :, 5)))

img_10_2 = rand(curng, obs_model_trO, 2)
@test size(img_10_2) == (100, 100, 10, 2)
@test eltype(img_10_2) == Float32
ℓ = @inferred logdensityof(obs_model_trO, img_10_2)
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
obs_model_TRO = obs_model_fn(T, R, O)
@test_throws DimensionMismatch img_10 = rand(curng, obs_model_TRO)

O = rand(curng, KernelUniform(0.0f0, 0.9f0), 100, 100, 10)
img_model_pose = obs_model_fn(T, R, O)
img_10 = rand(curng, img_model_pose)
@test size(img_10) == (100, 100, 10)
@test eltype(img_10) == Float32
ℓ = @inferred logdensityof(img_model_pose, img_10)
@test size(ℓ) == (10,)
@test ℓ isa CuArray{Float32}
for layer_id in 1:(size(img_10)[3]-1)
    @test @views img_10[:, :, layer_id] != img_10[:, :, layer_id+1]
    @test Array(ℓ)[layer_id] != Array(ℓ)[layer_id+1]
end
maybe_plot(plot_depth_img, Array(view(img_10, :, :, 8)))

img_10_2 = rand(curng, img_model_pose, 2)
@test size(img_10_2) == (100, 100, 10, 2)
@test eltype(img_10_2) == Float32
ℓ = @inferred logdensityof(img_model_pose, img_10_2)
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
