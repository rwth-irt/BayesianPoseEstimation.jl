# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

using Base.Broadcast: broadcasted
using Base: Callable
using SciGL

"""
    ObservationModel(pixel_dist, normalization_constant)
Model to compare rendered and observed depth images.

# Pixel Distribution
Each pixel is assumed to be independent and the measurement can be described by a distribution `pixel_dist(μ, o)`.
`μ` is the expected depth and `o` is the object association probability.
Therefore, the image logdensity for the measurement `z` is calculated by summing the pixel logdensities.
Other static parameters should be applied partially to the function beforehand (or worse be hardcoded).

# Normalization
If a normalization_constant is provided, `pixel_dist` is wrapped by a `ValidPixel`, which ignores invalid expected depth values (== 0).
This effectively changes the number of data points evaluated in the sum of the image loglikelihood for different views.
Thus, the algorithm might prefer (incorrect) poses further or closer to the object, which depends if the pixel loglikelihood is positive or negative.

To remove the sensitivity to the number of valid expected pixels, the images is normalized by diving the sum of the pixel loglikelihood by the number of valid pixels.
The `normalization_constant` is multiplied afterwards, a reasonable constant is the expected number of visible pixels for the views from the prior.

# Alternatives to Normalization
* proper preprocessing by cropping or segmenting the image
* a pixel_dist which handles the tail distribution by providing a reasonable likelihood for invalid expected values
"""
struct ObservationModel{normalized,P<:ManipulatedFunction,T}
    pixel_dist::P
    normalization_constant::T
end

ObservationModel(pixel_dist) = ObservationModel_(false, pixel_dist, 1)
ObservationModel(pixel_dist::ManipulatedFunction{names}, normalization_constant) where {names} = ObservationModel_(true, pixel_dist, normalization_constant)
# Hide this one, since normalization only makes sense if the constant is provided
ObservationModel_(normalized::Bool, pixel_dist::P, normalization_constant::T) where {P,T} = ObservationModel{normalized,P,T}(pixel_dist, normalization_constant)

# Image is always 2D
const Base.Dims(::Type{<:ObservationModel}) = (1, 2)
const Base.Dims(::ObservationModel) = Dims(ObservationModel)

"""
    realize(observation_model, sample)
The pixel_dist is parametrized by latent variables (μ, o, ...).
This function creates a BroadcastedDistribution which is conditioned on the matching variable names of the pixel_dist.
For a normalized observation_model, the pixel_dist is additionally wrapped in a ValidPixel.
"""
function realize(obs_model::ObservationModel{true}, sample::Sample)
    dist(μ, x...) = ValidPixel(μ, obs_model.pixel_dist(x...))
    BroadcastedDistribution(dist, Dims(ObservationModel), variables(sample).μ, values(variables(sample), obs_model.pixel_dist)...)
end

realize(obs_model::ObservationModel, sample::Sample) = BroadcastedDistribution(obs_model.pixel_dist, Dims(ObservationModel), values(variables(sample), obs_model.pixel_dist)...)

Base.values(nt::NamedTuple, ::ManipulatedFunction{names}) where {names} = values(nt[names])

"""
    rand(rng, observation_model, sample)
Generate a random observation using the expected depth μ and association o of the sample.
"""
Base.rand(rng::AbstractRNG, model::ObservationModel, sample::Sample) = rand(rng, realize(model, sample))

# DensityInterface
@inline DensityInterface.DensityKind(::ObservationModel) = HasDensity()

# Avoid ambiguities: extract variables μ, o & z from the sample
logdensityof_(model::ObservationModel, x::Sample) = logdensityof(realize(model, x), variables(x).z)

DensityInterface.logdensityof(model::ObservationModel, x::Sample) = logdensityof_(model, x)

function DensityInterface.logdensityof(model::ObservationModel{true}, x::Sample)
    log_p = logdensityof_(model, x)
    # Normalization: divide by the number of rendered pixels
    n_pixel = nonzero_pixels(variables(x).μ, Dims(model))
    model.normalization_constant ./ n_pixel .* log_p
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
struct ValidPixel{T<:Real,D} <: AbstractKernelDistribution{T,Continuous}
    # Should not cause memory overhead if used in lazily broadcasted context
    μ::T
    # Do not constrain M<:AbstractKernelDistribution{T} because it might be transformed / truncated
    model::D
end

function Distributions.logpdf(dist::ValidPixel{T}, x) where {T}
    # TODO Does the insupport dist(dist, x) help or not?
    if iszero(dist.μ)
        # If the expected value is invalid, it does not provide any information
        zero(T)
    else
        logdensityof(dist.model, clamp(x, minimum(dist), maximum(dist)))
    end
end

function Base.rand(rng::AbstractRNG, dist::ValidPixel{T}) where {T}
    if !insupport(dist, dist.μ)
        zero(T)
    else
        depth = rand(rng, dist.model)
        # maximum is inf
        clamp(depth, minimum(dist), maximum(dist))
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
