# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using CUDA
using MCMCDepth
using Plots
using Random
using SciGL
using Test

CUDA.allowscalar(false)
params = Parameters()
@reset params.float_type = Float32
@reset params.device = :CUDA

f_x = 1.2 * params.width
f_y = f_x
c_x = 0.5 * params.width
c_y = 0.5 * params.height
camera = CvCamera(params.width, params.height, f_x, f_y, c_x, c_y; near=params.min_depth, far=params.max_depth) |> Camera

gl_context = render_context(params)
cube_path = joinpath(dirname(pathof(SciGL)), "..", "examples", "meshes", "cube.obj")
model = upload_mesh(gl_context, cube_path)
scene = Scene(camera, [model])

cpu_rng = Random.default_rng()
Random.seed!(cpu_rng, params.seed)
dev_rng = device_rng(params)
Random.seed!(dev_rng, params.seed)

# PriorModel
# Pose only makes sense on CPU since CUDA cannot start render calls to OpenGL
gt_t = ([0.0f0, 0.0f0, 2.5f0])
t = BroadcastedNode(:t, cpu_rng, KernelNormal, gt_t, params.σ_t)
r = BroadcastedNode(:r, cpu_rng, QuaternionUniform, Float32)
# Using the CUDA rng requires the parameters to be stored in CuArrays
o = BroadcastedNode(:o, dev_rng, KernelUniform, CUDA.fill(0.0f0, params.width, params.height), CUDA.fill(1.0f0, params.width, params.height))

@testset "PosteriorModel" begin
    prior = (t=t, r=r, o=o)
    sample = @inferred rand(prior)
    @test sample isa NamedTuple{(:t, :r, :o)}
    @test sample.t isa Array{Float32}
    @test size(sample.t) == (3,)
    @test sample.r isa Quaternion{Float32}
    @test sample.o isa CuArray{Float32}
    @test size(sample.o) == (params.width, params.height)
    ℓ = @inferred logdensityof(prior, sample)
    @test ℓ isa Float32

    sample = @inferred rand(prior, 5)
    @test sample isa NamedTuple{(:t, :r, :o)}
    @test sample.t isa Array{Float32}
    @test size(sample.t) == (3, 5)
    @test sample.r isa Array{Quaternion{Float32}}
    @test size(sample.r) == (5,)
    @test sample.o isa CuArray{Float32}
    @test size(sample.o) == (params.width, params.height, 5)
    ℓ = @inferred logdensityof(prior, sample)
    @test ℓ isa Array{Float32}
    @test size(ℓ) == (5,)


    # PosteriorModel
    μ_fn = render_fn | (gl_context, scene)
    μ = DeterministicNode(:μ, μ_fn, (; t=t, r=r))
    pixel = pixel_valid_mixture | (params.min_depth, params.max_depth, params.pixel_θ, params.pixel_σ)
    z = BroadcastedNode(:z, dev_rng, pixel, (; μ=μ, o=o))
    z_norm = ModifierNode(z, dev_rng, ImageLikelihoodNormalizer | params.normalization_constant)
    depth_img = rand(z_norm).z
    posterior = PosteriorModel(z_norm, (; z=depth_img))

    sample = @inferred rand(posterior, 5)
    ℓ = @inferred logdensityof(posterior, sample)
    sample = @inferred rand(posterior)
    ℓ = @inferred logdensityof(posterior, sample)
end

# plot_depth_img(depth_img)
# plot_depth_img(sample.variables.μ)
