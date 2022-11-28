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
dev_rng = CUDA.default_rng()
Random.seed!(dev_rng, 42)
CUDA.allowscalar(false)

# Parameters
render_context = RenderContext(100, 100, 50, CuArray)
scene = Scene(Parameters(), render_context)
t = [0, 0, 1.5]
r = Quaternion(1, 0, 0, 0) |> normalize
μ_img = render_fn(render_context, scene, 1, t, r)
prior_o = 0.8f0
pixel_σ = 0.1f0
pixel_θ = 1.0f0
min_depth = 0.1f0
max_depth = 2.0f0
obs = rand(dev_rng, pixel_mixture.(min_depth, max_depth, pixel_θ, pixel_θ, μ_img, prior_o))

# PixelAssociation
dist_is = valid_pixel_normal | pixel_σ
dist_not = valid_pixel_tail | (min_depth, max_depth, pixel_θ)
pix_ass = pixel_association | (dist_is, dist_not, prior_o)


# Test pixel formula
z = 1.0f0
μ = 1.01f0
o = @inferred pix_ass(μ, z)
@test o == prior_o * pdf(dist_is(μ), z) / (prior_o * pdf(dist_is(μ), z) + (1 - prior_o) * pdf(dist_not(μ), z))


@inferred logdensityof(pa, 0.1f0)
@inferred logdensityof(pa, 0.0f0)
@inferred logdensityof(pa, 1.0f0)
@test_throws DomainError logdensityof(pa, -eps(Float32))
@test_throws DomainError logdensityof(pa, 1.0f0 + eps(Float32))

# Invalid z are clamped in ValidPixel 
@test pix_ass(1.0f0, 0.0f0) == pix_ass(1.0f0, -eps(Float32)) == pix_ass(1.0f0, max_depth) == pix_ass(1.0f0, max_depth + eps(Float32))
# negative example
@test pix_ass(-eps(Float32), 1.0f0) != pix_ass(eps(Float32), 1.0f0)

# Invalid μ should result in the prior
@test pix_ass(0.0f0, 1.0f0) == prior_o
@test pix_ass(-eps(Float32), 1.0f0) == prior_o
@test pix_ass(max_depth + eps(Float32), 1.0f0) != prior_o
# Negative examples
@test pix_ass(eps(Float32), 1.0f0) != prior_o
@test pix_ass(max_depth, 1.0f0) != prior_o

# Likely to be associated
prior_range = 0:0.01:1
μ = 1.0f0
z = 0.99f0
maybe_plot(plot, prior_range, pixel_association.(dist_is, dist_not, prior_range, μ, z), label="p(o|μ=$μ,z=$z,oₚᵣᵢₒᵣ)", xlabel="oₚᵣᵢₒᵣ")
# Unlinkely to be associated
z = 0.7f0
maybe_plot(plot!, prior_range, pixel_association.(dist_is, dist_not, prior_range, μ, z), label="p(o|μ=$μ,z=$z,oₚᵣᵢₒᵣ)", xlabel="oₚᵣᵢₒᵣ")

# ImageAssociation for pixel-wise association
ia = MCMCDepth.image_association(dist_is, dist_not, prior_o, obs, :o, :μ)
s = @inferred rand(ia, (; μ=μ_img))
# Requires other variables to be present
@test_throws ErrorException rand(ia)

# Should all be prior_o for invalid μ
s = @inferred rand(ia, (; μ=CUDA.fill(0.0f0, size(obs))))
@test reduce(&, s.o .== prior_o)
# Should not be prior_o for valid μ
s = @inferred rand(ia, (; μ=μ_img, distractor=1))
@test !reduce(&, s.o .== prior_o)
maybe_plot(plot_depth_img, Array(s.o))

# Should not contribute to logdensityof
ℓ = @inferred logdensityof(ia, s)
@test ℓ == 0