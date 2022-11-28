# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

using Random
using SciGL

"""
    ImageLikelihoodNormalizer
Use it in a modifier node to normalize the loglikelihood of the image to make it independent from the number of visible pixels in μ. 
"""
struct ImageLikelihoodNormalizer{T<:Real,M<:AbstractArray{T}}
    normalization_constant::T
    μ::M
end

ImageLikelihoodNormalizer(normalization_constant::T, μ::M, _...) where {T,M} = ImageLikelihoodNormalizer{T,M}(normalization_constant, μ)

Base.rand(::AbstractRNG, ::ImageLikelihoodNormalizer, value) = value
using DensityInterface
function DensityInterface.logdensityof(model::ImageLikelihoodNormalizer, z, ℓ)
    # Images are always 2D
    n_pixel = sum_and_dropdims(model.μ .!= 0, (1, 2))
    ℓ .* model.normalization_constant ./ n_pixel
end

"""
    expected_pixel_count(rng, prior_model, render_context, scene, parameters)
Calculates the expected number of valid rendered pixels for the poses of the prior model.
This number can for example be used as the normalization constant in the observation model.
"""
function expected_pixel_count(rng::AbstractRNG, prior_model, render_context::RenderContext, scene::Scene, parameters::Parameters)
    n_pixel = Vector{parameters.precision}(undef, 0)
    for _ in 1:cld(parameters.n_normalization_samples, parameters.depth)
        prior_sample = rand(rng, prior_model, parameters.depth)
        img = render(render_context, scene, parameters.object_id, to_pose(variables(prior_sample).t, variables(prior_sample).r))
        append!(n_pixel, nonzero_pixels(img, (1, 2)))
    end
    mean(n_pixel)
end

"""
    pixel_mixture(min_depth, max_depth, θ, σ, μ, o)
Mixture distribution for a depth pixel: normal / tail.
The mixture is weighted by the association o for the normal and 1-o for the tail.

* Normal distribution: measuring the object of interest with the expected depth μ and standard deviation σ
* Tail distribution: occlusions (exponential) and random outliers (uniform)
"""
function pixel_mixture(min_depth::T, max_depth::T, θ::T, σ::T, μ::T, o::T) where {T<:Real}
    normal = KernelNormal(μ, σ)
    tail = pixel_tail(min_depth, max_depth, θ, μ)
    KernelBinaryMixture(normal, tail, o, one(o) - o)
end

valid_pixel_mixture(min_depth::T, max_depth::T, θ::T, σ::T, μ::T, o::T) where {T<:Real} = ValidPixel(μ, pixel_mixture(min_depth, max_depth, θ, σ, μ, o))

function pixel_tail(min_depth::T, max_depth::T, θ::T, μ::T) where {T<:Real}
    # TODO Does truncated make a difference? Should effectively do the same as checking for valid pixel, since the logdensity will be 0 for μ ⋜ min_depth
    # exponential = KernelExponential
    # truncated must satisfy: lower <= upper → max(min_depth, μ)
    # TODO what about the μ in the association model?
    exponential = truncated(KernelExponential(θ), min_depth, max(min_depth, μ))
    uniform = KernelUniform(zero(T), max_depth)
    # TODO custom weights for exponential and uniform?
    KernelBinaryMixture(exponential, uniform, one(T), one(T))
end

valid_pixel_tail(min_depth::T, max_depth::T, θ::T, μ::T) where {T<:Real} = ValidPixel(μ, pixel_tail(min_depth, max_depth, θ, μ))

pixel_normal(σ::T, μ::T) where {T<:Real} = KernelNormal(μ, σ)
valid_pixel_normal(σ, μ) = ValidPixel(μ, KernelNormal(μ, σ))


"""
    pixel_explicit(min_depth, max_depth, θ, σ, μ, o)
Mixture distribution for a depth pixel which explicitly handles invalid μ.
In case the expected depth is invalid, only the tail distribution for outliers is evaluated.
Otherwise, if the measured depth and expected depth are zero, a unreasonably high likelihood would be returned.
The mixture is weighted by the association o for the normal and 1-o for the tail.

* Normal distribution: measuring the object of interest with the expected depth μ and standard deviation σ
* Tail distribution: occlusions (exponential) and random outliers (uniform)
"""
function pixel_explicit(min_depth::T, max_depth::T, θ::T, σ::T, μ::T, o::T) where {T<:Real}
    if μ > 0
        pixel_mixture(min_depth, max_depth, θ, σ, μ, o)
    else
        # Distribution must be of same type for CUDA support so set o to zero to evaluate the tail only
        pixel_mixture(min_depth, max_depth, θ, σ, μ, zero(T))
    end
end

valid_pixel_explicit(min_depth::T, max_depth::T, θ::T, σ::T, μ::T, o::T) where {T<:Real} = ValidPixel(μ, pixel_explicit(min_depth, max_depth, θ, σ, μ, o))

"""
    render_fn(render_context, scene, object_id, t, r)
Function can be conditioned on the render_context, scene & object_id to be used in a model node to render different poses for t & r.
"""
function render_fn(render_context, scene, object_id, t, r)
    p = to_pose(t, r)
    render(render_context, scene, object_id, p)
end