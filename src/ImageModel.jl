# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# TEST

# TODO from Sample? I think PoseModel should handle it to hide the internal μ from the chain

"""
    ImageModel(pixel_dist, μ, o, normalize)
Model to compare rendered and observed depth images.

# TODO Could I infer θ instead of o analytically, too? Integration might be possible for exponential family and conjugate priors. However, I would need to keep o fixed.
During inference it takes care of missing values in the expected depth `μ` and only evaluates the logdensity for pixels with depth 0 < z < max_depth.
Invalid values of z are set to zero by convention.

Each pixel is assumed to be independent and the measurement can be described by a distribution `pixel_dist(μ, o)`.
`μ` is the expected value and `o` is the object association probability.
Other static parameters should be applied partially to the function beforehand (or worse be hardcoded).
"""
struct ImageModel{T,U<:AbstractArray,O<:AbstractArray}
    pixel_dist::T
    μ::U
    o::O
    normalize::Bool
end

"""
    ImageModel(pixel_dist, μ, θ, o)
Make sure that the order of the parameters of pixel_dist matches μ, θ, o.
"""
ImageModel(pixel_dist, μ, o) = ImageModel(pixel_dist, μ, o, true)

BroadcastedDistribution(model::ImageModel) = BroadcastedDistribution(model.pixel_dist, Dims(model), model.μ, model.o)

# Image is always 2D
Base.Dims(::ImageModel) = (1, 2)
Base.Dims(::Type{<:ImageModel}) = (1, 2)

# Generate independent random numbers from m_pix(μ, o)
Base.rand(rng::AbstractRNG, model::ImageModel, dims...) = rand(rng, BroadcastedDistribution(model), dims...)

function DensityInterface.logdensityof(model::ImageModel, x)
    log_p = logdensityof(BroadcastedDistribution(model), x)
    if model.normalize
        # Count the number of rendered pixels and divide by it
        rendered_pixels = sum_and_dropdims(model.μ .> 0; dims=Dims(model))
        return log_p ./ rendered_pixels
    end
    # no normalization = raw sum of the pixel likelihoods
    log_p
end

# TODO Custom indices more efficient? Possible on GPU without allocations?
# function DensityInterface.logdensityof(d::ImageModel, z)
#     # Only sum the logdensity for values for which filter_fn is true
#     ind = findall(x -> d.params.min_depth < x < d.params.max_depth, d.μ)
#     sum = 0.0
#     for i in ind
#         # Preprocess the measurement
#         z_i = d.params.min_depth < z[i] < d.params.max_depth ? z[i] : zero(z[i])
#         sum = sum + logdensity(d.params.pixel_measure(d.μ[i], d.o[i], d.params), z_i)
#     end
#     sum
# end

"""
    PixelDistribution
Distribution of an independent pixel which handles out of range measurements by ignoring them.
"""
struct PixelDistribution{T<:Real,U} <: AbstractKernelDistribution{T,Continuous}
    min::T
    max::T
    model::U
end

# TODO move distribution generator function to main script / specific experiment script. Best practice: one script per experiment?
# WARN named parameters are less efficient
"""
    pixel_normal_exponential(σ, min, max, μ, θ, o)
Generate a Pixel distribution from the given parameters.
Putting static parameters first allows partial application of the function.
"""
function pixel_normal_exponential(σ, θ, min, max, μ, o)
    dist = KernelBinaryMixture(KernelNormal(μ, σ), KernelExponential(θ), o, 1.0 - o)
    PixelDistribution(min, max, dist)
end
pixel_normal_exponential_default = pixel_normal_exponential | (0.1, 0.1, 3)

# Handle invalid values by ignoring them (log probability is zero)
Distributions.logpdf(dist::PixelDistribution{T}, x) where {T} = insupport(dist, x) ? logdensityof(dist.model, x) : zero(T)

function Base.rand(rng::AbstractRNG, dist::PixelDistribution{T}) where {T}
    depth = rand(rng, dist.model)
    insupport(dist, depth) ? depth : zero(T)
end

Base.maximum(dist::PixelDistribution) = dist.max
Base.minimum(dist::PixelDistribution) = dist.min
# logpdf handles out of support so no transformation is required or even desired
Bijectors.bijector(dist::PixelDistribution) = Bijectors.Identity{0}()
Distributions.insupport(dist::PixelDistribution, x::Real) = minimum(dist) < x < maximum(dist)
