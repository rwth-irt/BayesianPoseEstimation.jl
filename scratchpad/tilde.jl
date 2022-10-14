using Accessors
using CUDA
using DensityInterface
using Distributions
using MCMCDepth
using Random
using Tilde

# TODO Constructing the whole thing is more eager than waiting for a sample in the functions

struct EagerObservation{normalized,T,M<:AbstractArray{T},D<:BroadcastedDistribution} <: Distribution{ArrayLikeVariate{2},Continuous}
    normalization_constant::T
    μ::M
    dist::D
end
EagerObservation(normalized::Bool, norm_const, μ::M, dist::D) where {T,M<:AbstractArray{T},D} = EagerObservation{normalized,T,M,D}(T(norm_const), μ, dist)

function EagerObservation(μ, pixel_dist, args...)
    dist = BroadcastedDistribution(pixel_dist, Dims(ObservationModel), args...)
    EagerObservation(false, μ, 1, dist)
end

function NormalizedLazyObservation(normalization_constant, μ, pixel_dist, args...)
    dist = BroadcastedDistribution(pixel_dist, Dims(ObservationModel), args...)
    EagerObservation(true, normalization_constant, μ, dist)
end

const Base.Dims(::Type{<:EagerObservation}) = (1, 2)
const Base.Dims(::EagerObservation) = Dims(ObservationModel)

Base.rand(rng::AbstractRNG, model::EagerObservation) = rand(rng, model.dist)

# DensityInterface
@inline DensityInterface.DensityKind(::EagerObservation) = HasDensity()

# Avoid ambiguities: extract variables μ, o & z from the sample
DensityInterface.logdensityof(model::EagerObservation, x) = logdensityof(model.dist, x)

function DensityInterface.logdensityof(model::EagerObservation{true}, x)
    log_p = logdensityof(model.dist, x)
    # Normalization: divide by the number of rendered pixels
    n_pixel = nonzero_pixels(model.μ, Dims(model))
    model.normalization_constant ./ n_pixel .* log_p
end

using MeasureBase

struct KernelMeasure{D}
    dist::D
end
Base.rand(rng::AbstractRNG, ::Type, m::KernelMeasure, dims...) = rand(rng, m.dist, dims...)
DensityInterface.logdensityof(m::KernelMeasure, x) = logdensityof(m.dist, x)

Tilde.predict(::AbstractRNG, ::KernelMeasure, pars) = pars
Tilde.predict(::AbstractRNG, pars) = pars
Tilde.predict(rng::AbstractRNG, m::KernelMeasure) = rand(rng, m)

# Override behavior so the correct rng is used
MeasureBase.AbstractMeasure(d::MCMCDepth.AbstractKernelDistribution) = KernelMeasure(d)
MeasureBase.AbstractMeasure(d::MCMCDepth.BroadcastedDistribution) = KernelMeasure(d)
MeasureBase.AbstractMeasure(d::EagerObservation) = KernelMeasure(d)

##### Using Tilde for flexible modelling ########
# TODO broadcast the sum of logdensities in Tilde's logdensity.jl

m = @model begin
    a ~ ProductBroadcastedDistribution(KernelExponential, CUDA.fill(1.0f0, 3, 3))
    μ = a * 2
    z ~ NormalizedLazyObservation(3, μ, KernelNormal, μ, 0.1f0)
end

s = rand(CUDA.default_rng(), m())
logdensityof(m(), s)
s1 = @set s.a = CUDA.rand(3, 3, 2)
logdensityof(m(), s1)
