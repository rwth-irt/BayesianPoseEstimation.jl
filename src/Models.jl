# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using AbstractMCMC
using MeasureTheory
using Random

"""
Standard deviation of the sensor noise.
"""
const PIX_σ = 0.1
"""
Expected value of the exponential distribution → Occlusion expected closer or further away.
"""
const PIX_θ = 1.0
"""
Range limit of the sensor
"""
const MAX_DEPTH = 10.0
"""
Mixture coefficient of the exponential part of the mixture.
"""
const MIX_EXPONENTIAL = 0.8
"""
Mixture coefficient of the uniform part of the mixture.
"""
const MIX_UNIFORM = 1.0 - MIX_EXPONENTIAL

"""
  DepthNormal(μ)
Normal distribution intended for observing the expected object.
Given the expected depth `μ`.
"""
DepthNormal(μ) = Normal(μ, PIX_σ)

"""
  DepthExponential(μ, o)
Exponential distribution intended for observing an occlusion.
Given the expected depth `μ`.
"""
DepthExponential() = Exponential(PIX_θ)

"""
  DepthUniform(μ, o)
Uniform distribution intended for observing random outliers.
Given the expected depth `μ`.
"""
DepthUniform() = UniformInterval(0, MAX_DEPTH)

"""
  DepthExponentialUniform(μ, o)
Mixture of exponential and uniform distribution intended for observing an occlusion or random outlier.
Given the expected depth `μ`.
"""
DepthExponentialUniform() = BinaryMixture(DepthExponential(), DepthUniform(), MIX_EXPONENTIAL, MIX_UNIFORM)

"""
  DepthNormalExponential(μ, o)
Assumes a normal distribution for the object and an uniform distribution for random outliers.
Given the expected depth `μ` and object association probability `o`.
"""
DepthNormalExponential(μ, o) = BinaryMixture(DepthNormal(μ), DepthExponential(), o, 1.0 - o)

"""
  DepthNormalUniform(μ, o)
Assumes a normal distribution for the object and an exponential distribution for occlusions.
Given the expected depth `μ` and object association probability `o`.
"""
DepthNormalUniform(μ, o) = BinaryMixture(DepthNormal(μ), DepthUniform(), o, 1.0 - o)

"""
  DepthNormalExponentialUniform(μ, o)
Assumes a normal distribution for the object and a mixture of an exponential and uniform distribution for occlusions and outliers.
Given the expected depth `μ` and object association probability `o`.
"""
DepthNormalExponentialUniform(μ, o) = BinaryMixture(DepthNormal(μ), DepthExponentialUniform(), o, 1.0 - o)

"""
  PosteriorModel
Models the posterior logdensity p(θ|y)~ℓ(y|θ)q(θ) up to a constant.
`q` is the prior model and should support a rand(q) and logdensity(q, θ).
`ℓ` is the observation model / likelihood for a sample.
"""
struct PosteriorModel <: AbstractMCMC.AbstractModel
  # Do not constrain types, only logdensity(..., θ) required
  q
  ℓ
end

"""
  logdensity(m, s)
Non-corrected logdensity of the of the sample `s` given the measure `m`.
"""
MeasureTheory.logdensity(m::PosteriorModel, s::Sample) =
  logdensity(m.q, s) + logdensity(m.ℓ, s)

"""
  pose_depth_model(mean_t, cov_t, obs_dist)
The probabilistic model of observing an object with a pose `r, t` in a depth image `z`.
Provide a observation model `obs_dist(μ, o)` for each pixel measurement given an expected depth `μ` and object association probability `o`.
"""
pose_depth_model = @model mean_t, cov_t, width, height, render_fn, obs_dist begin
  t ~ MvNormal(mean_t, cov_t)
  # We don't know anything about the orientation or the occlusion
  r .~ fill(CircularUniform(), 3)
  o .~ fill(Uniform(), width, height)
  μ = render_fn(t, r)
  # TODO how to preprocess z in here
  z .~ obs_dist.(μ, o)
end


"""
  DepthImageMeasure(μ, o, m_pix, filter_fn, prep_fn)
Optimized measure for handling observation of depth images.
When doing inference it takes care of missing values in the expected depth `μ` and only evaluates the logdensity for pixels which for which `filter_fn(z_i)=true`.
Before evaluating the logdensity of a pixel, it is preprocessed using `prep_fn`.

Each pixel is assumed to be independent and the measurement can be described by the measure `m_pix(μ, o)`.
`o` is the object association probability.

For the generative model, the whole image is generated.
"""
struct DepthImageMeasure{M<:Base.Callable,F<:Base.Callable,P<:Base.Callable} <: AbstractMeasure
  μ::Matrix{Float64}
  o::Matrix{Float64}
  m_pix::M
  filter_fn::F
  prep_fn::P
end
MeasureTheory.basemeasure(::DepthImageMeasure) = Lebesgue{ℝ₊}
TransformVariables.as(d::DepthImageMeasure) = as(Array, asℝ₊, size(d.μ))

Base.show(io::IO, d::DepthImageMeasure) = print(io, "DepthImageMeasure\n  Pixel measure: $(d.m_pix)\n  Filter function: $(d.filter_fn)\n  Preprocessing function: $(d.prep_fn)")

# Generate independent random numbers from m_pix(μ, o)
Base.rand(rng::AbstractRNG, T::Type, d::DepthImageMeasure) = [rand(rng, T, d.m_pix(x...)) for x in zip(d.μ, d.o)]

function MeasureTheory.logdensity(d::DepthImageMeasure, z)
  # Only sum the logdensity for values for which filter_fn is true
  ind = findall(d.filter_fn, d.μ)
  sum = 0.0
  for i in ind
    # Possibly preprocess the measurement
    z_i = d.prep_fn(z[i])
    println(z_i)
    sum = sum + logdensity(d.m_pix(d.μ[i], d.o[i]), z_i)
  end
  sum
end


"""
  pixel_association(μ, z, q, p_is, p_not)
Probability of the pixel being associated to the object.
Given an expected depth of `μ` and observation of `z` the posterior is calculated using Bayes Law with the prior `q`.
The distribution of observing the object is constructed via `d_is(μ)` and other observations are explained by `d_not(μ)`
"""
function pixel_association(μ, z, q, d_is, d_not)::Float64
  # If the rendered value is invalid, we do not know more than before => prior
  if μ <= 0.0
    return q
  end
  prior_likelihood = pdf(d_is(μ), z) * q
  # Marginalize Bernoulli distributed by summing out o
  marginal = prior_likelihood + pdf(d_not(), z) * (1 - q)
  # Normalized posterior
  posterior = prior_likelihood / marginal
  # robust for transformation as𝕀 ∈ (0,1), might get numerically 0.0 or 1.0
  if posterior < eps()
    return eps()
  elseif posterior > 1 - eps()
    return 1 - eps()
  else
    return posterior
  end
end

"""
  image_association(s, z, prior_o)
Renders the state of the sample `s` using the `render_fn` and then calls `pixel_association` with `z, q, d_is, d_not` on each rendered pixel.
"""
function image_association(s::Sample, z, q, d_is, d_not, render_fn)
  t, r = state(s).t, state(s).r
  μ = render_fn(t, r)
  o = pixel_association.(μ, z, q, d_is, d_not)
  tr = as((; o = as(Array, as𝕀, size(o))))
  Sample((; o = o), -Inf, tr)
end

"""
  nonzero_indices(img)
Returns a list of indices for the nonzero pixels in the image.
"""
nonzero_indices(img) = findall(!iszero, img)

# TODO remove
"""
  _preprocess(render, img)
Reduce the computational load by operating only on the `indices`.
Recommendation: `nonzero_indices` to extract the non zero indices of the rendered image.
"""
function _preprocess(indices, img::AbstractArray)
  # Use only the nonzero values of the render for the likelihood
  view_img = view(img, indices)
  map(view_img) do x
    # Convention for invalid values: depth=0
    if 0 < x < MAX_DEPTH
      x
    else
      zero(x)
    end
  end
end

"""
  preprocess(render, μ, o, z)
Reduce the computational load by operating only on the nonzero `indices` of `μ`.
Recommendation: `nonzero_indices` to extract the non zero indices of the rendered image.
"""
function preprocess(μ::AbstractArray{T,N}, other::AbstractArray{T,N}...) where {T,N}
  ind = nonzero_indices(μ)
  res = Vector{Vector{T}}()
  push!(res, _preprocess(ind, μ))
  for o in other
    push!(res, _preprocess(ind, o))
  end
  res
end