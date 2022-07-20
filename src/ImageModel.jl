# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# TEST
# TODO is there any more elegant way? ObservationModel generates ImageModel on the fly which generates the PixelModel on the fly. This requires passing image_parameters down to the PixelModel. Probably only a Monolith model would solve this which generates the distributions in the functions.

"""
    ObservationModel
Provide a position `t`, orientation `r` and occlusion `o`.
Sets the pose of [object_id] to render depth of the scene.
This is used in the logdensityof with the pixel_dist to compare the expected and the measured images.
"""
struct ObservationModel{C<:RenderContext,S<:Scene,D,O<:AbstractArray,P<:AbstractVector{<:Pose}}
    # Latent variables last to enable partial application using FunctionManipulation.jl
    render_context::C
    scene::S
    object_id::Int
    pixel_dist::D
    normalize_img::Bool
    occlusion::O
    poses::P
end

"""
    ObservationModel(parameters, render_context, t, r, o)
Convenience constructor which extracts the parameters into the ObservationModel and converts raw array representations of the position and orientation to a vector of poses.
"""
function ObservationModel(parameters::Parameters, render_context::RenderContext, t, r, o)
    poses = to_pose(t, r, parameters.rotation_type)
    ObservationModel(render_context, parameters.scene, parameters.object_id, parameters.pixel_dist, parameters.normalize_img, o, poses)
end

render(model::ObservationModel) = render(model.context, model.scene, model.object_id, model.t, model.r)

"""
    rand(rng, model, dims...)
Generate a sample with variables (t=position, r=orientation, o=occlusion).
"""
function Base.rand(rng::AbstractRNG, model::ObservationModel, dims...)
    img_model = ImageModel(model)
    # TEST dims applies noise multiple times?
    rand(rng, img_model, dims...)
end

function DensityInterface.logdensityof(model::ObservationModel, x)
    img_model = ImageModel(model)
    logdensityof(img_model, x)
end

"""
    ImageModel(obs_model)
Conveniently construct the image model from an ObservationModel.
"""
function ImageModel(obs_model::ObservationModel)
    # μ as internal variable which will not be part of the sample
    μ = render(obs_model)
    ImageModel(obs_model.pixel_dist, μ, obs_model.o, obs_model.normalize_img)
end

# TODO Could I infer θ instead of o analytically, too? Integration might be possible for exponential family and conjugate priors. However, I would need to keep o fixed.

"""
    ImageModel(pixel_dist, μ, o, normalize)
Model to compare rendered and observed depth images.
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
    BroadcastedDistribution(img_model)
Broadcast the expected value μ and the occlusion to generate pixel level distributions for all the latent variables.
Since images are always 2D, the BroadcastedDistribution reduction dims are fixed to (1,2).
"""
BroadcastedDistribution(img_model::ImageModel) = BroadcastedDistribution(img_model.pixel_dist, Dims(img_model), img_model.μ, img_model.o)

# Image is always 2D
const Base.Dims(::Type{<:ImageModel}) = (1, 2)
const Base.Dims(::ImageModel) = Dims(ImageModel)

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

# TODO Custom indices more efficient? Possible on GPU without allocations? Should be handled by insupport? Branching in Kernels usually is not wanted.
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
