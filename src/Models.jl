# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

using DensityInterface
using LinearAlgebra

"""
# ObservationModel
Model to compare rendered and observed depth images.

# Pixel Distribution
Each pixel is assumed to be independent and the measurement can be described by a distribution `pixel_dist(μ, o)`.
`μ` is the expected depth and `o` is the object association probability.
Therefore, the image logdensity for the measurement `z` is calculated by summing the pixel logdensities.
Other static parameters should be applied partially to the function beforehand (or worse be hardcoded).

# Normalization
If `c_reg` is provided, `pixel_dist` is wrapped by a `ValidPixel`, which ignores invalid expected depth values (== 0).
This effectively changes the number of data points evaluated in the sum of the image loglikelihood for different views.
Thus, the algorithm might prefer (incorrect) poses further or closer to the object, which depends if the pixel loglikelihood is positive or negative.

To remove the sensitivity to the number of valid expected pixels, the images is normalized by diving the sum of the pixel loglikelihood by the number of valid pixels.
The `c_reg` is multiplied afterwards.

# Alternatives to Normalization
* proper preprocessing by cropping or segmenting the image
* a pixel_dist which handles the tail distribution by providing a reasonable likelihood for invalid expected values
"""

"""
    ValidPixel
Takes care of missing values in the expected depth `μ == 0` by setting the logdensity to zero, effectively ignoring these pixels in the sum.
Consequently, the sum of the image likelihood must be normalized by dividing through the number of valid pixels, since the likelihood is very sensitive to the number of evaluated data points.
"""
struct ValidPixel{T<:Real,D} <: AbstractKernelDistribution{T,Continuous}
    # Should not cause memory overhead if used in lazily broadcasted context
    μ::T
    # Do not constrain M<:AbstractKernelDistribution{T} because it might be transformed / truncated
    model::D
end

function Distributions.logpdf(dist::ValidPixel{T}, x) where {T}
    if !insupport(dist, dist.μ)
        # If the expected value is invalid, it does not provide any information
        zero(T)
    else
        # clamp to avoid NaNs
        logdensityof(dist.model, clamp(x, minimum(dist), maximum(dist)))
    end
end

function KernelDistributions.rand_kernel(rng::AbstractRNG, dist::ValidPixel{T}) where {T}
    if !insupport(dist, dist.μ)
        zero(T)
    else
        depth = rand(rng, dist.model)
        # maximum is inf
        clamp(depth, minimum(dist), maximum(dist))
    end
end

# Depth pixels can have any positive value not like radar
Base.maximum(dist::ValidPixel{T}) where {T} = maximum(dist.model)
# Negative measurements do not make any sense, all others might, depending on the underlying model.
Base.minimum(dist::ValidPixel{T}) where {T} = max(zero(T), minimum(dist.model))
# logpdf explicitly handles outliers, so no transformation is desired
Bijectors.bijector(dist::ValidPixel) = Bijectors.TruncatedBijector(minimum(dist), maximum(dist))
Distributions
# Depth pixels can have any positive value, zero and negative are invalid
Distributions.insupport(dist::ValidPixel, x::Real) = minimum(dist) < x

########## Likelihood normalization ##########

"""
    ImageLikelihoodNormalizer
Serves as a regularization of the image likelihood, where the independence assumption of individual pixels does not hold.
Use it in a modifier node to normalize the loglikelihood of the image to make it less dependent from the number of visible pixels in μ.

ℓ_reg = c_reg / n_visible_pixel * ℓ
"""
struct ImageLikelihoodNormalizer{T<:Real,M<:AbstractArray{T}}
    c_reg::T
    μ::M
    # NOTE using the prior_o instead of the estimated o is worse
    o::M
end

ImageLikelihoodNormalizer(c_reg::T, μ::M, _...) where {T,M} = ImageLikelihoodNormalizer{T,M}(c_reg, μ)

Base.rand(::AbstractRNG, ::ImageLikelihoodNormalizer, value) = value
function DensityInterface.logdensityof(model::ImageLikelihoodNormalizer, z, ℓ)
    # NOTE This incentives poses where only a handful of pixels is visible at the edges of the image which perfectly fit the measured depth. Especially for non distinct features.
    # n_μ = model.μ != 0
    # logdensity_npixel.(ℓ,  model.c_reg, n_μ)

    # NOTE This regularization incentives a minimization of the visible pixels, e.g. fitting the silhouette into the prior mask. - loglikelihood grows more than linear with the number of pixels? Pixel association does not modify the prior in regions where nothing is rendered.
    # union = @. model.μ != 0 || model.o > 0.5
    # n_pixel = sum_and_dropdims(union, (1, 2))
    # logdensity_npixel.(ℓ, model.c_reg, n_pixel)

    # NOTE this is more stable than the above and should still capture the varying number of pixels. It fuses the information form the prior and the observation so it is the best guess of pixels which actually contribute information on the pose.
    n_o = sum_and_dropdims(model.o .>= 0.5, (1, 2))
    logdensity_npixel.(ℓ, model.c_reg, n_o)
end
"""
    logdensity_npixel(ℓ, c_reg, n_pixel)
(Broadcastable) Avoid undefined behavior for n_pixel = 0 - CPU: x/0=Inf, CUDA x/0=NaN.

Nothing visible is very unlikely so return -∞ as loglikelihood.

Otherwise returns: c_reg / n_pixel * ℓ
"""
logdensity_npixel(ℓ, c_reg, n_pixel) = iszero(n_pixel) ? typemin(ℓ) : c_reg / n_pixel * ℓ

# TODO evaluate this in diss. Isn't it in Probabilistic Robotics? :D Introduce Hyperparameter like in the more complex one?
"""
    SimpleImageRegularization
Use it in a modifier node to regularize the loglikelihood of the image to make it less dominant compared to the prior.
Tunable Hyperparameter: c_reg is the regularization constant.

ℓ_reg = c_reg / n_pixel * ℓ
"""
struct SimpleImageRegularization
    c_reg
end
SimpleImageRegularization(c_reg, _...) = SimpleImageRegularization(c_reg)

Base.rand(::AbstractRNG, ::SimpleImageRegularization, value) = value
DensityInterface.logdensityof(model::SimpleImageRegularization, z, ℓ) = logdensity_npixel(ℓ, model.c_reg, length(z))

"""
    expected_pixel_count(rng, prior_model, render_context, scene, parameters)
Calculates the expected number of valid rendered pixels for the poses of the prior model.
This number can for example be used as the normalization constant in the observation model.
"""
function expected_pixel_count(rng::AbstractRNG, prior_model, render_context::OffscreenContext, scene::Scene, parameters::Parameters)
    n_pixel = Vector{parameters.float_type}(undef, 0)
    for _ in 1:cld(parameters.n_normalization_samples, parameters.depth)
        prior_sample = rand(rng, prior_model, parameters.depth)
        img = render(render_context, scene, parameters.object_id, to_pose(variables(prior_sample).t, variables(prior_sample).r))
        append!(n_pixel, nonzero_pixels(img, (1, 2)))
    end
    mean(n_pixel)
end

"""
    nonzero_pixels(images, dims)
Calculates the number of nonzero pixels for each image with the given dims.
"""
nonzero_pixels(images, dims) = sum_and_dropdims(images .!= 0, dims)

######### Pixel models #########

"""
    pixel_mixture(min_depth, max_depth, θ, σ, μ, o)
Mixture distribution for a depth pixel: normal / tail.
The mixture is weighted by the association o for the normal and 1-o for the tail.

* Normal distribution: measuring the object of interest with the expected depth μ and standard deviation σ
* Tail distribution: occlusions (exponential) and random outliers (uniform)
"""
function pixel_mixture(min_depth::T, max_depth::T, θ::T, σ::T, μ::T, o::T) where {T<:Real}
    normal = KernelNormal(μ, σ)
    # NOTE Exponential in pixel_tail does actually seems to be beneficial under heavy occlusions
    tail = pixel_tail(min_depth, max_depth, θ, σ, μ)
    BinaryMixture(normal, tail, o, one(o) - o)
end

pixel_valid_mixture(min_depth::T, max_depth::T, θ::T, σ::T, μ::T, o::T) where {T<:Real} = ValidPixel(μ, pixel_mixture(min_depth, max_depth, θ, σ, μ, o))

function pixel_tail(min_depth::T, max_depth::T, θ::T, σ::T, μ::T) where {T<:Real}
    # NOTE Truncated does not seem to make a difference. Should effectively do the same as checking for valid pixel, since the logdensity will be 0 for μ ⋜ min_depth. Here, a smooth Exponential is beneficial which avoids 0.
    exponential = KernelExponential(θ)
    uniform = TailUniform(min_depth, max_depth)
    # TODO custom weights for exponential and uniform?
    BinaryMixture(exponential, uniform, one(T), one(T))
end

pixel_valid_tail(min_depth::T, max_depth::T, θ::T, σ::T, μ::T) where {T<:Real} = ValidPixel(μ, pixel_tail(min_depth, max_depth, θ, σ, μ))

"""
    smooth_mixture(min_depth, max_depth, θ, σ, μ, o)
Mixture distribution for a depth pixel: normal / tail.
The mixture is weighted by the association o for the normal and 1-o for the tail.

* Normal distribution: measuring the object of interest with the expected depth μ and standard deviation σ
* Tail distribution: occlusions are modeled by a SmoothExponential distribution and random outliers via a TailUniform
"""
function smooth_mixture(min_depth::T, max_depth::T, θ::T, σ::T, μ::T, o::T) where {T<:Real}
    normal = KernelNormal(μ, σ)
    tail = smooth_tail(min_depth, max_depth, θ, σ, μ)
    BinaryMixture(normal, tail, o, one(o) - o)
end

smooth_valid_mixture(min_depth::T, max_depth::T, θ::T, σ::T, μ::T, o::T) where {T<:Real} = ValidPixel(μ, smooth_mixture(min_depth, max_depth, θ, σ, μ, o))

function smooth_tail(min_depth::T, max_depth::T, θ::T, σ::T, μ::T) where {T<:Real}
    exponential = SmoothExponential(min_depth, μ, θ, σ)
    uniform = TailUniform(min_depth, max_depth)
    # TODO custom weights for exponential and uniform?
    BinaryMixture(exponential, uniform, one(T), one(T))
end

smooth_valid_tail(min_depth::T, max_depth::T, θ::T, σ::T, μ::T) where {T<:Real} = ValidPixel(μ, smooth_tail(min_depth, max_depth, θ, σ, μ))

pixel_normal(σ::T, μ::T) where {T<:Real} = KernelNormal(μ, σ)
pixel_valid_normal(σ, μ) = ValidPixel(μ, KernelNormal(μ, σ))

pixel_valid_uniform(min_depth, max_depth, μ) = ValidPixel(μ, TailUniform(min_depth, max_depth))

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
        # Distribution must be of same type for type stable CUDA support so set o to zero to evaluate the tail only
        pixel_mixture(min_depth, max_depth, θ, σ, μ, zero(T))
    end
end

pixel_valid_explicit(min_depth::T, max_depth::T, θ::T, σ::T, μ::T, o::T) where {T<:Real} = ValidPixel(μ, pixel_explicit(min_depth, max_depth, θ, σ, μ, o))

"""
    render_fn(render_context, scene, object_id, t, r)
Function can be conditioned on the render_context, scene & object_id to be used in a model node to render different poses for t & r.
"""
function render_fn(render_context, scene, t, r)
    p = to_pose(t, r)
    render(render_context, scene, p)
end

function position_prior(params, experiment, rng)
    t = BroadcastedNode(:t, rng, KernelNormal, experiment.prior_t, params.σ_t)
    r = BroadcastedNode(:r, rng, QuaternionUniform, params.float_type)
    (; t=t, r=r)
end

function μ_model(render_context, experiment, prior)
    μ_fn = render_fn | (render_context, experiment.scene)
    μ = DeterministicNode(:μ, μ_fn, (prior.t, prior.r))
    (; t=prior.t, r=prior.r, μ=μ)
end

"""
    marginalized_association(dist_is, dist_not, prior, μ, z)
Consists of a distribution `dist_is(μ)` for the probability of a pixel belonging to the object of interest and `dist_not(μ)` which models the probability of the pixel not belonging to this object.
Moreover, a `prior` is required for the association probability `o`.
The `logdensityof` the observation `z` is calculated analytically by marginalizing the two distributions.
"""
function marginalized_association(dist_is, dist_not, prior, μ, z)
    # Internal ValidPixels handle outliers by returning 1.0 as probability which will result in the prior q without too much overhead
    p_is = pdf(dist_is(μ), z)
    p_not = pdf(dist_not(μ), z)
    nominator = prior * p_is
    # Marginalize Bernoulli distributed by summing out o
    marginal = nominator + (1 - prior) * p_not
    # Normalized posterior
    nominator / marginal
end

"""
    pixel_association_fn(params)
Returns a function `fn(prior, μ, z)` which analytically calculates the association probability via marginalization.
Uses:
* normal distribution for measuring the object of interest.
* uniform distribution for the tail, i.e. measuring anything but the object of interest.
"""
function pixel_association_fn(params)
    dist_is = pixel_valid_normal | params.association_σ
    dist_not = pixel_valid_tail | (params.min_depth, params.max_depth, params.pixel_θ, params.association_σ)
    marginalized_association | (dist_is, dist_not)
end

"""
    smooth_association_fn(params)
Returns a function `fn(prior, μ, z)` which analytically calculates the association probability via marginalization.
Uses:
* normal distribution for measuring the object of interest.
* mixture of a smoothly truncated exponential and uniform distribution for the tail, i.e. measuring anything but the object of interest.
"""
function smooth_association_fn(params)
    dist_is = pixel_valid_normal | params.association_σ
    dist_not = smooth_valid_tail | (params.min_depth, params.max_depth, params.pixel_θ, params.association_σ)
    marginalized_association | (dist_is, dist_not)
end
