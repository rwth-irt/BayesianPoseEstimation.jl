# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

using Random
using SciGL

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
    pixel_mixture(min_depth, max_depth, σ, θ, μ, o)
Mixture distribution for a depth pixel: normal / tail.
The mixture is weighted by the association o for the normal and 1-o for the tail.

* Normal distribution: measuring the object of interest with the expected depth μ and standard deviation σ
* Tail distribution: occlusions (exponential) and random outliers (uniform)
"""
function pixel_mixture(min_depth::T, max_depth::T, σ::T, θ::T, μ::T, o::T) where {T<:Real}
    # TODO Does truncated make a difference?
    # truncate: lower <= upper → max(min_depth, μ)
    exponential = truncated(KernelExponential(θ), min_depth, max(min_depth, μ))
    uniform = KernelUniform(zero(T), max_depth)
    # TODO custom weights for exponential and uniform?
    tail = KernelBinaryMixture(exponential, uniform, one(T), one(T))
    normal = KernelNormal(μ, σ)
    tail = KernelBinaryMixture(exponential, uniform, one(T), one(T))
    KernelBinaryMixture(normal, tail, o, one(o) - o)
end

"""
    pixel_explicit(min_depth, max_depth, σ, θ, μ, o)
Mixture distribution for a depth pixel which explicitly handles invalid μ.
In case the expected depth is invalid, only the tail distribution for outliers is evaluated.
Otherwise, if the measured depth and expected depth are zero, a unreasonably high likelihood would be returned.
The mixture is weighted by the association o for the normal and 1-o for the tail.

* Normal distribution: measuring the object of interest with the expected depth μ and standard deviation σ
* Tail distribution: occlusions (exponential) and random outliers (uniform)
"""
function pixel_explicit(min_depth::T, max_depth::T, σ::T, θ::T, μ::T, o::T) where {T<:Real}
    if μ > 0
        pixel_mixture(min_depth, max_depth, σ, θ, μ, o)
    else
        # Distribution must be of same type for CUDA support so set o to zero to evaluate the tail only
        pixel_mixture(min_depth, max_depth, σ, θ, μ, o)
        # pixel_mixture(zero(T), max_depth, σ, θ, μ, zero(T))
    end
end