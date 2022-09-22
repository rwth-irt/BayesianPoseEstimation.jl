# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Base.Broadcast: broadcasted
using Base: Callable
using SciGL

"""
    ObservationModel(normalize_img, broadcasted_dist, μ)
Model to compare rendered and observed depth images.
During inference it takes care of missing values in the expected depth `μ` and only evaluates the logdensity for pixels with depth 0 < z < max_depth.
Invalid values of z are set to zero by convention.

Each pixel is assumed to be independent and the measurement can be described by a distribution `pixel_dist(μ, o)`.
`μ` is the expected value and `o` is the object association probability.
Other static parameters should be applied partially to the function beforehand (or worse be hardcoded).
"""
struct ObservationModel{T,B<:BroadcastedDistribution{T},U<:AbstractArray{T}}
    normalize_img::Bool
    broadcasted_dist::B
    # For the calculation of the normalization constant
    μ::U
end

"""
    ObservationModel(normalize_img, pixel_dist, μ, o)
Generates a `BroadcastedDistribution` of dim (1,2) from the `pix_dist`, expected depth `μ` and the association probability `o`.
"""
function ObservationModel(normalize_img::Bool, pixel_dist, μ::AbstractArray, o)
    broadcasted_dist = BroadcastedDistribution(pixel_dist, Dims(ObservationModel), μ, o)
    ObservationModel(normalize_img, broadcasted_dist, μ)
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
        rendered_pixels = sum_and_dropdims(model.μ .> 0; dims=Dims(model))
        return log_p ./ rendered_pixels
    end
    # no normalization = raw sum of the pixel likelihoods
    log_p
end

"""
    PixelDistribution
Distribution of an independent pixel which handles invalid expected values `μ`
"""
struct PixelDistribution{T<:Real,M} <: AbstractKernelDistribution{T,Continuous}
    # Shouls not cause memory overhead if used in lazily broadcasted context
    μ::T
    # Do not constrain M<:AbstractKernelDistribution{T} because it might be transformed / truncated
    model::M
end

function Distributions.logpdf(dist::PixelDistribution{T}, x) where {T}
    if insupport(dist, dist.μ) && insupport(dist, x)
        logdensityof(dist.model, x)
    else
        # If the expected value or the observation is invalid, it does not provide any information
        zero(T)
    end
end

function Base.rand(rng::AbstractRNG, dist::PixelDistribution{T}) where {T}
    if !insupport(dist, dist.μ)
        zero(T)
    else
        depth = rand(rng, dist.model)
        # maximum is inf
        depth > minimum(dist) ? depth : minimum(dist)
    end
end

# Depth pixels can have any positive value not like radar
Base.maximum(::PixelDistribution{T}) where {T} = typemax(T)
# Negative measurements do not make any sense, all others might, depending on the underlying model.
Base.minimum(dist::PixelDistribution{T}) where {T} = max(zero(T), minimum(dist.model))
# logpdf explicitly handles outliers, so no transformation is desired
Bijectors.bijector(::PixelDistribution) = Bijectors.TruncatedBijector(minimum(dist), maximum(dist))
Distributions
# Depth pixels can have any positive value, zero and negative are invalid
Distributions.insupport(dist::PixelDistribution, x::Real) = minimum(dist) < x
