# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Base.Broadcast: broadcasted
using Base: Callable
using SciGL

"""
    ImageModel(normalize_img, broadcasted_dist, μ)
Model to compare rendered and observed depth images.
During inference it takes care of missing values in the expected depth `μ` and only evaluates the logdensity for pixels with depth 0 < z < max_depth.
Invalid values of z are set to zero by convention.

Each pixel is assumed to be independent and the measurement can be described by a distribution `pixel_dist(μ, o)`.
`μ` is the expected value and `o` is the object association probability.
Other static parameters should be applied partially to the function beforehand (or worse be hardcoded).
"""
struct ImageModel{T,N,B<:BroadcastedDistribution{T,N},U<:AbstractArray{T}}
    normalize_img::Bool
    broadcasted_dist::B
    # Need to access μ to calculate the normalization constant
    μ::U
end

"""
    ImageModel(normalize_img, pixel_dist, μ, o)
Generates a `BroadcastedDistribution` of dim (1,2) from the `pix_dist`, expected depth `μ` and the association probability `o`.
"""
function ImageModel(normalize_img::Bool, pixel_dist, μ::AbstractArray, o::AbstractArray)
    broadcasted_dist = BroadcastedDistribution(pixel_dist, Dims(ImageModel), μ, o)
    ImageModel(normalize_img, broadcasted_dist, μ)
end

"""
    ImageModel(render_context, scene, object_id, rotation_type, normalize_img, pixel_dist, t, r, o)
Generate an ImageModel by rendering the expected depth `μ` for the provided scene.
Sets the pose of `object_id` to the position(s) `t` and orientation(s) `r`.
"""
function ImageModel(render_context::RenderContext, scene::Scene, object_id::Integer, rotation_type::Type, normalize_img::Bool, pixel_dist, t::AbstractArray, r::AbstractArray, o::AbstractArray)
    p = to_pose(t, r, rotation_type)
    μ = render(render_context, scene, object_id, p)
    ImageModel(normalize_img, pixel_dist, μ, o)
end

"""
    ImageModel(parameters, render_context, scene, t, r, o)
Convenience constructor which extracts the `object_id`, `normalize_img`, `rotation_type` and `pixel_dist` from the `parameters` struct.
Note that `rotation_type` and `pixel_dist` are expected as Symbols which get evaled at runtime. 
"""
ImageModel(parameters::Parameters, render_context::RenderContext, scene::Scene, t::AbstractArray, r::AbstractArray, o::AbstractArray) = ImageModel(render_context, scene, parameters.object_id, eval(parameters.rotation_type), parameters.normalize_img, eval(parameters.pixel_dist), t, r, o)

# Image is always 2D
const Base.Dims(::Type{<:ImageModel}) = (1, 2)
const Base.Dims(::ImageModel) = Dims(ImageModel)

# Generate independent random numbers from m_pix(μ, o)
Base.rand(rng::AbstractRNG, model::ImageModel, dims::Integer...) = rand(rng, model.broadcasted_dist, dims...)

# DensityInterface
@inline DensityKind(::ImageModel) = HasDensity()

function DensityInterface.logdensityof(model::ImageModel, x)
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
Distribution of an independent pixel which handles out of range measurements by ignoring them.
"""
struct PixelDistribution{T<:Real,M} <: AbstractKernelDistribution{T,Continuous}
    min::T
    max::T
    model::M
end

# Handle invalid values by ignoring them (log probability is zero)
Distributions.logpdf(dist::PixelDistribution{T}, x) where {T} = insupport(dist, x) ? logdensityof(dist.model, x) : zero(T)

function Base.rand(rng::AbstractRNG, dist::PixelDistribution{T}) where {T}
    depth = rand(rng, dist.model)
    insupport(dist, depth) ? depth : zero(T)
end

Base.maximum(dist::PixelDistribution) = dist.max
Base.minimum(dist::PixelDistribution) = dist.min
# logpdf explicitly handles outliers, so no transformation is desired
Bijectors.bijector(::PixelDistribution) = Bijectors.Identity{0}()
Distributions.insupport(dist::PixelDistribution, x::Real) = minimum(dist) < x < maximum(dist)
