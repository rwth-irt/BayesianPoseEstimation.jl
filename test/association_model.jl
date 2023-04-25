# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using LinearAlgebra
using MCMCDepth
using Plots
using Random
using SciGL
using Test

# WARN OpenGL & CUDA interop not trivially possible in CI
rng = Random.default_rng()
Random.seed!(rng, 42)

# Parameters
params = Parameters()
f_x = 1.2 * params.width
f_y = f_x
c_x = 0.5 * params.width
c_y = 0.5 * params.height
camera = CvCamera(params.width, params.height, f_x, f_y, c_x, c_y; near=params.min_depth, far=params.max_depth) |> Camera

gl_context = depth_offscreen_context(params.width, params.height, params.depth, Array)
cube_path = joinpath(dirname(pathof(SciGL)), "..", "examples", "meshes", "cube.obj")
model = upload_mesh(gl_context, cube_path)
scene = Scene(camera, [model])
t = [0, 0, 1.5]
r = Quaternion(1, 0, 0, 0)
μ_img = render_fn(gl_context, scene, t, r)
prior_o = 0.8f0
pixel_σ = 0.1f0
pixel_θ = 1.0f0
min_depth = 0.1f0
max_depth = 2.0f0
pixel_dist = pixel_mixture | (min_depth, max_depth, pixel_θ, pixel_θ)
img_dist = BroadcastedDistribution(pixel_dist, μ_img, prior_o)
obs = @inferred rand(rng, img_dist)

# PixelAssociation
dist_is = pixel_valid_normal | pixel_σ
dist_not = pixel_valid_tail | (min_depth, max_depth, pixel_θ, pixel_σ)
pix_ass = MCMCDepth.marginalized_association | (dist_is, dist_not, prior_o)

@testset "Pixel association" begin
    # Test pixel formula
    z = 1.0f0
    μ = 1.01f0
    o = @inferred pix_ass(μ, z)
    @test o == prior_o * pdf(dist_is(μ), z) / (prior_o * pdf(dist_is(μ), z) + (1 - prior_o) * pdf(dist_not(μ), z))

    # Invalid μ should result in the prior
    @test pix_ass(0.0f0, 1.0f0) == prior_o
    @test pix_ass(-eps(Float32), 1.0f0) == prior_o
    @test pix_ass(max_depth + eps(Float32), 1.0f0) != prior_o
    # Negative examples
    @test pix_ass(eps(Float32), 1.0f0) != prior_o
    @test pix_ass(max_depth, 1.0f0) != prior_o
end

# Visual validation
# Likely to be associated
# prior_range = 0:0.01:1
# μ = 1.0f0
# z = 0.99f0
# plot(prior_range, MCMCDepth.marginalized_association.(dist_is, dist_not, prior_range, μ, z), label="p(o|μ=$μ,z=$z,oₚᵣᵢₒᵣ)", xlabel="oₚᵣᵢₒᵣ")
# # Unlinkely to be associated
# z = 0.7f0
# plot!(prior_range, MCMCDepth.marginalized_association.(dist_is, dist_not, prior_range, μ, z), label="p(o|μ=$μ,z=$z,oₚᵣᵢₒᵣ)", xlabel="oₚᵣᵢₒᵣ")
