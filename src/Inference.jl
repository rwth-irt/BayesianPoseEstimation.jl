# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

using Random
using SciGL

"""
    pose_prior(render_context, scene, object_id, t_model, r_model, o_model)
Creates a PriorModel for the variables t, r & o which automatically renders μ after in rand.
"""
pose_prior(render_context::RenderContext, scene::Scene, object_id::Integer, t_model, r_model, o_model) = RenderModel(render_context, scene, object_id, IndependentModel((; t=t_model, r=r_model, o=o_model))) |> PriorModel(model, bijectors)


"""
    post
Consist of a `prior_model`, which generates a sample with variables t, r, o & μ.
The `observation_model` is a function of (μ, o) which creates a ObservationModel.
The observation itself is the `z` variable of the sample.
"""
pose_posterior(render_context::RenderContext, scene::Scene, object_id::Integer, t_model, r_model, o_model, z_model) = PosteriorModel(prior, ConditionedModel(likelihood, (; z=observation)))

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
    pixel_mixture(min_depth, max_depth, θ, σ; μ, o)
Mixture distribution for a depth pixel: normal / tail.
The mixture is weighted by the association o for the normal and 1-o for the tail.

* Normal distribution: measuring the object of interest with the expected depth μ and standard deviation σ
* Tail distribution: occlusions (exponential) and random outliers (uniform)
"""
function pixel_mixture(min_depth::T, max_depth::T, θ::T, σ::T; μ::T, o::T) where {T<:Real}
    normal = KernelNormal(μ, σ)
    tail = pixel_tail(min_depth, max_depth, θ, μ)
    KernelBinaryMixture(normal, tail, o, one(o) - o)
end

# TODO autogenerate (:μ, :o) via PPL?
"""
    pixel_mixture(parameters)
Generates a pixel_mixture(μ, o) distribution given the parameters.
"""
pixel_mixture(parameters::Parameters) = pixel_mixture | (:μ, :o) | (parameters.min_depth, parameters.max_depth, parameters.pixel_θ, parameters.pixel_σ)

function pixel_tail(min_depth::T, max_depth::T, θ::T, μ::T) where {T<:Real}
    # TODO Does truncated make a difference?
    # truncate: lower <= upper → max(min_depth, μ)
    # TODO what about the μ in the association model?
    exponential = truncated(KernelExponential(θ), min_depth, max(min_depth, μ))
    uniform = KernelUniform(zero(T), max_depth)
    # TODO custom weights for exponential and uniform?
    KernelBinaryMixture(exponential, uniform, one(T), one(T))
end

"""
    pixel_explicit(min_depth, max_depth, θ, σ; μ, o)
Mixture distribution for a depth pixel which explicitly handles invalid μ.
In case the expected depth is invalid, only the tail distribution for outliers is evaluated.
Otherwise, if the measured depth and expected depth are zero, a unreasonably high likelihood would be returned.
The mixture is weighted by the association o for the normal and 1-o for the tail.

* Normal distribution: measuring the object of interest with the expected depth μ and standard deviation σ
* Tail distribution: occlusions (exponential) and random outliers (uniform)
"""
function pixel_explicit(min_depth::T, max_depth::T, θ::T, σ::T; μ::T, o::T) where {T<:Real}
    if μ > 0
        pixel_mixture(min_depth, max_depth, θ, σ; μ=μ, o=o)
    else
        # Distribution must be of same type for CUDA support so set o to zero to evaluate the tail only
        pixel_mixture(min_depth, max_depth, θ, σ; μ=μ, o=zero(T))
    end
end

# TODO autogenerate (:μ, :o) via PPL?
"""
    pixel_explicit(parameters)
Generates a pixel_explicit(μ, o) distribution given the parameters.
"""
pixel_explicit(parameters::Parameters) = pixel_explicit | (:μ, :o) | (parameters.min_depth, parameters.max_depth, parameters.pixel_θ, parameters.pixel_σ)
