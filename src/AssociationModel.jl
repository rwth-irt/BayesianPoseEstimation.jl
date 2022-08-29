# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using DensityInterface
using Distributions

# TODO should a Sample be passed?

"""
    ImageAssociation
Broadcasts the analytic PixelAssociation over images given the expected depth `μ` and a prior association `prior`.
Each PixelAssociation consists of a distribution `dist_is` for the probability of a pixel belonging to the object of interest and `dist_is` which models the probability of the pixel not belonging to this object.
"""
struct ImageAssociation{T,N,B<:BroadcastedDistribution{T,N}}
    broadcasted_dist::B
end

function ImageAssociation(dist_is, dist_not, dims, prior, μ)
    # Make PixelAssociation broadcastable over q and μ
    pixel_association(q, μ) = PixelAssociation(q, dist_is(μ), dist_not(μ))
    ImageAssociation(BroadcastedDistribution(pixel_association, dims, prior, μ))
end

# General case is not reduction → dims=()
ImageAssociation(dist_is, dist_not, prior, μ) = ImageAssociation(dist_is, dist_not, (), prior, μ)
# In case of scalar prior, the output of the BroadcastedDistribution should also be a scalar per image → dims=(1,2)
ImageAssociation(dist_is, dist_not, prior::Real, μ) = ImageAssociation(dist_is, dist_not, (1, 2), prior, μ)

const Base.Dims(::ImageAssociation{<:Any,N}) where {N} = Dims(N)

# Sample from p(o|q,μ)
Base.rand(rng::AbstractRNG, model::ImageAssociation, dims::Integer...) = rand(rng, model.broadcasted_dist, dims...)

@inline DensityKind(::ImageAssociation) = HasDensity()
# Represents log(p(o|q,μ,z))
DensityInterface.logdensityof(model::ImageAssociation, x) = logdensityof(model.broadcasted_dist, x)

# Per pixel association

"""
    PixelAssociation
Consists of a distribution `dist_is` for the probability of a pixel belonging to the object of interest and `dist_is` which models the probability of the pixel not belonging to this object.
Moreover, a prior `prior` is required for the association probability.
Typically both distributions are conditioned on the expected depth μ of the pixel.
The logpdf is calculated analytically by marginalizing the two distributions.
"""
struct PixelAssociation{T,I<:PixelDistribution,N<:PixelDistribution} <: AbstractKernelDistribution{T,Continuous}
    prior::T
    dist_is::I
    dist_not::N
end

function Distributions.logpdf(dist::PixelAssociation, x)
    # Internal PixelDistributions handle outliers by returning 1.0 as probability which will result in the prior q without too much overhead
    p_is = pdf(dist.dist_is, x)
    p_not = pdf(dist.dist_not, x)
    nominator = dist.prior * p_is
    # Marginalize Bernoulli distributed by summing out o
    marginal = nominator + (1 - dist.prior) * p_not
    # Normalized posterior
    nominator / marginal
end

function Base.rand(rng::AbstractRNG, dist::PixelAssociation)
    # Sample from the prior
    mix = KernelBinaryMixture(dist.dist_is, dist.dist_not, dist.prior, (1 - dist.prior))
    rand(rng, mix)
end

# The support of a mixture is the union of the support of its components
Base.maximum(dist::PixelAssociation) = max(maximum(dist.dist_is), maximum(dist.dist_not))
Base.minimum(dist::PixelAssociation) = min(minimum(dist.dist_is), minimum(dist.dist_not))
# logpdf of the PixelDistributions explicitly handles outliers, so no transformation is desired
Bijectors.bijector(::PixelAssociation) = Bijectors.Identity{0}()
Distributions.insupport(dist::PixelAssociation, x::Real) = minimum(dist) < x < maximum(dist)
