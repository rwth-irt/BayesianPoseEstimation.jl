# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

using Base.Broadcast: broadcasted
using Base: Callable
using SciGL

"""
    ObservationModel(normalize_img, normalization_constant, broadcasted_dist, μ)
Model to compare rendered and observed depth images.
Accumulates and optionally normalizes the pixel distribution.

# Pixel Distribution
Each pixel is assumed to be independent and the measurement can be described by a distribution `pixel_dist(μ, o)`.
`μ` is the expected depth and `o` is the object association probability.
Other static parameters should be applied partially to the function beforehand (or worse be hardcoded).

# Normalization
The `pixel_dist` is wrapped by a `ValidPixel`, which ignores invalid expected depth values (== 0).
This effectively changes the number of data points evaluated in the sum of the image loglikelihood for different views.
Thus, the algorithm might prefer (incorrect) poses further or closer to the object, which depends if the pixel loglikelihood is positive or negative.

To remove the sensitivity to the number of valid expected pixels, the images is normalized by diving the sum of the pixel loglikelihood by the number of valid pixels.
The `normalization_constant` is multiplied afterwards, a reasonable constant is the expected number of visible pixels for the views from the prior.

# Alternatives to Normalization
* proper preprocessing by cropping or segmenting the image
* a pixel_dist which handles the tail distribution by providing a reasonable likelihood for invalid expected values

"""
struct ObservationModel{T,B<:BroadcastedDistribution{T},U<:AbstractArray{T}}
    normalize_img::Bool
    normalization_constant::T
    broadcasted_dist::B
    # For the calculation of the normalization constant
    μ::U
end

"""
    ObservationModel(normalize_img, pixel_dist, μ, o)
Generates a `BroadcastedDistribution` of dim (1,2) from the `pix_dist`, expected depth `μ` and the association probability `o`.
"""
function ObservationModel(normalization_constant::T, pixel_dist, μ::AbstractArray{T}, o::Union{T,AbstractArray{T}}) where {T}
    wrapped_dist(pμ, po) = ValidPixel(pμ, pixel_dist(pμ, po))
    broadcasted_dist = BroadcastedDistribution(wrapped_dist, Dims(ObservationModel), μ, o)
    ObservationModel(true, normalization_constant, broadcasted_dist, μ)
end

# TODO Doc the different constructors
# TEST different behavior
function ObservationModel(pixel_dist, μ::AbstractArray{T}, o::Union{T,AbstractArray{T}}) where {T}
    broadcasted_dist = BroadcastedDistribution(pixel_dist, Dims(ObservationModel), μ, o)
    ObservationModel(false, one(T), broadcasted_dist, μ)
end

# Image is always 2D
const Base.Dims(::Type{<:ObservationModel}) = (1, 2)
const Base.Dims(::ObservationModel) = Dims(ObservationModel)

# Generate independent random numbers from m_pix(μ, o)
Base.rand(rng::AbstractRNG, model::ObservationModel, dims::Integer...) = rand(rng, model.broadcasted_dist, dims...)

# DensityInterface
@inline DensityKind(::ObservationModel) = HasDensity()

function DensityInterface.logdensityof(model::ObservationModel, x)
    log_p = logdensityof(model.broadcasted_dist, x)
    if model.normalize_img
        # Normalization: divide by the number of rendered pixels
        n_pixel = nonzero_pixels(model.μ, Dims(model))
        return model.normalization_constant / n_pixel * log_p
    end
    # no normalization = raw sum of the pixel likelihoods
    log_p
end

"""
    nonzero_pixels(images, dims)
Calculates the number of nonzero pixels for each image with the given dims.
"""
nonzero_pixels(images, dims) = sum_and_dropdims(images .!= 0, dims)

"""
    ValidPixel
Takes care of missing values in the expected depth `μ == 0` by setting the logdensity to zero, effectively ignoring these pixels in the sum.
Consequently, the sum of the image likelihood must be normalized by dividing through the number of valid pixels, since the likelihood is very sensitive to the number of evaluated data points.
"""
struct ValidPixel{T<:Real,M} <: AbstractKernelDistribution{T,Continuous}
    # Shouls not cause memory overhead if used in lazily broadcasted context
    μ::T
    # Do not constrain M<:AbstractKernelDistribution{T} because it might be transformed / truncated
    model::M
end

function Distributions.logpdf(dist::ValidPixel{T}, x) where {T}
    # TODO Does the insupport dist(dist, x) help or not?
    if iszero(dist.μ)
        # If the expected value is invalid, it does not provide any information
        zero(T)
    else
        logdensityof(dist.model, x)
    end
end

function Base.rand(rng::AbstractRNG, dist::ValidPixel{T}) where {T}
    if !insupport(dist, dist.μ)
        zero(T)
    else
        depth = rand(rng, dist.model)
        # maximum is inf
        depth > minimum(dist) ? depth : minimum(dist)
    end
end

# Depth pixels can have any positive value not like radar
Base.maximum(::ValidPixel{T}) where {T} = typemax(T)
# Negative measurements do not make any sense, all others might, depending on the underlying model.
Base.minimum(dist::ValidPixel{T}) where {T} = max(zero(T), minimum(dist.model))
# logpdf explicitly handles outliers, so no transformation is desired
Bijectors.bijector(::ValidPixel) = Bijectors.TruncatedBijector(minimum(dist), maximum(dist))
Distributions
# Depth pixels can have any positive value, zero and negative are invalid
Distributions.insupport(dist::ValidPixel, x::Real) = minimum(dist) < x
