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

**Invalid values of z** (the measurement) must be preprocessed an set to inf.

# Regularization
Previously, `ValidPixel` has been used to determine whether the rendered image was in the support of the distributions.
Now, the distributions must handle values outside their support by returning a logdensity of -Inf.
WARN: Truncated distributions return +Inf if min=max=z. 

The algorithm might prefer (incorrect) poses further or closer to the object, which depends if the pixel loglikelihood is positive or negative.
To remove the sensitivity to the number of valid expected pixels, the images is **normalized** by diving the sum of the pixel loglikelihood by the number of valid pixels.
The `c_reg` is multiplied afterwards.

Alternatives include:
* proper preprocessing by cropping or segmenting the image
* a pixel_dist which handles the tail distribution by providing a reasonable likelihood for invalid expected values
"""



########## Likelihood normalization ##########

"""
    ImageLikelihoodNormalizer
Serves as a regularization of the image likelihood, where the independence assumption of individual pixels does not hold.
Use it in a modifier node to normalize the loglikelihood of the image to make it less dependent from the number of visible pixels in μ.

NOTE: Seems to be beneficial for occlusions.

ℓ_reg = c_reg / n_visible_pixel * ℓ
"""
struct ImageLikelihoodNormalizer{T<:Real,M<:AbstractArray{T},N<:AbstractArray{T}}
    c_reg::T
    μ::M
    # NOTE using the prior_o instead of the estimated o is worse
    o::N
end

ImageLikelihoodNormalizer(c_reg::T, μ::M, _...) where {T,M} = ImageLikelihoodNormalizer{T,M}(c_reg, μ)

Base.rand(::AbstractRNG, ::ImageLikelihoodNormalizer, value) = value
function DensityInterface.logdensityof(model::ImageLikelihoodNormalizer, z, ℓ)
    # Must be >0.5, >=0.5 would include the indifferent 50:50 prior
    # n_o = sum_and_dropdims(model.o .> 0.5, (1, 2))
    n_o = sum_and_dropdims(model.o, (1, 2))
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
DensityInterface.logdensityof(model::SimpleImageRegularization, z, ℓ) = logdensity_npixel.(ℓ, model.c_reg, length(z))

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

function pixel_tail(min_depth::T, max_depth::T, θ::T, σ::T, μ::T) where {T<:Real}
    exponential = KernelExponential(θ)
    uniform = TailUniform(min_depth, max_depth)
    # TODO custom weights for exponential and uniform?
    BinaryMixture(exponential, uniform, one(T), one(T))
end

"""
    truncated_mixture(min_depth, max_depth, θ, σ, μ, o)
Mixture distribution for a depth pixel: normal / tail.
The mixture is weighted by the association o for the normal and 1-o for the tail.

* Normal distribution: measuring the object of interest with the expected depth μ and standard deviation σ
* Tail distribution: occlusions are modeled by a truncated Exponential distribution and random outliers via a TailUniform
"""
function truncated_mixture(min_depth::T, max_depth::T, θ::T, σ::T, μ::T, o::T) where {T<:Real}
    normal = KernelNormal(μ, σ)
    tail = truncated_tail(min_depth, max_depth, θ, σ, μ)
    BinaryMixture(normal, tail, o, one(o) - o)
end

# NOTE seems to be beneficial for occlusions
function truncated_tail(min_depth::T, max_depth::T, θ::T, σ::T, μ::T) where {T<:Real}
    if μ > 0
        exponential = truncated(KernelExponential(θ), nothing, μ)
    else
        # if μ==0 && z==0, Inf is returned for the logdensity which breaks everything
        # must return the same type for CUDA so truncated without limits
        # only uniform should be returned, β=Inf → logdensity=-Inf
        exponential = truncated(KernelExponential(typemax(T)), nothing, nothing)
    end
    uniform = TailUniform(min_depth, max_depth)
    # TODO custom weights for exponential and uniform?
    BinaryMixture(exponential, uniform, one(T), one(T))
end

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

function smooth_tail(min_depth::T, max_depth::T, θ::T, σ::T, μ::T) where {T<:Real}
    # Occlusions might occur in front of min_depth
    exponential = SmoothExponential(zero(T), μ, θ, σ)
    uniform = TailUniform(min_depth, max_depth)
    # TODO custom weights for exponential and uniform?
    BinaryMixture(exponential, uniform, one(T), one(T))
end

pixel_normal(σ::T, μ::T) where {T<:Real} = KernelNormal(μ, σ)

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

Assumes that dist_not covers ℝ and returns a nonzero probability.
Limit cases: prior==1 → 1, prior==0 → 0
"""
function marginalized_association(dist_is, dist_not, prior, μ, z)
    # Return limit if no update is possible - avoids division by zero
    if iszero(prior) || isone(prior) || iszero(μ)
        return prior
    end
    p_is = pdf(dist_is(μ), z)
    p_not = pdf(dist_not(μ), z)
    nominator = prior * p_is
    # Marginalize Bernoulli distributed by summing out o
    marginal = nominator + (1 - prior) * p_not
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
    dist_is = pixel_normal | params.association_σ
    dist_not = pixel_tail | (params.min_depth, params.max_depth, params.pixel_θ, params.association_σ)
    marginalized_association | (dist_is, dist_not)
end

"""
    truncated_association_fn(params)
Returns a function `fn(prior, μ, z)` which analytically calculates the association probability via marginalization.
Uses:
* normal distribution for measuring the object of interest.
* mixture of a truncated exponential and uniform distribution for the tail, i.e. measuring anything but the object of interest.
"""
function truncated_association_fn(params)
    dist_is = pixel_normal | params.association_σ
    dist_not = truncated_tail | (params.min_depth, params.max_depth, params.pixel_θ, params.association_σ)
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
    dist_is = pixel_normal | params.association_σ
    dist_not = smooth_tail | (params.min_depth, params.max_depth, params.pixel_θ, params.association_σ)
    marginalized_association | (dist_is, dist_not)
end
