# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using AbstractMCMC
using MeasureTheory, Soss
using Random

"""
  DepthNormal(μ, p)
Normal distribution intended for observing the expected object.
Given the expected depth `μ`.
"""
DepthNormal(μ, p::DepthImageParameters) = Normal(μ, p.pix_σ)

"""
  DepthExponential(p)
Exponential distribution intended for observing an occlusion.
"""
DepthExponential(p::DepthImageParameters) = Exponential(p.pix_θ)

"""
  DepthUniform(p)
Uniform distribution intended for observing random outliers.
"""
DepthUniform(p::DepthImageParameters) = UniformInterval(p.min_depth, p.max_depth)

"""
  DepthExponentialUniform(p)
Mixture of exponential and uniform distribution intended for observing an occlusion or random outlier.
"""
DepthExponentialUniform(p::DepthImageParameters) = BinaryMixture(DepthExponential(p), DepthUniform(p), p.mix_exponential, 1 - p.mix_exponential)

"""
  DepthNormalExponential(μ, o, p)
Assumes a normal distribution for the object and an uniform distribution for random outliers.
Given the expected depth `μ` and object association probability `o`.
"""
DepthNormalExponential(μ, o, p::DepthImageParameters) = BinaryMixture(DepthNormal(μ, p), DepthExponential(p), o, 1.0 - o)

"""
  DepthNormalUniform(μ, o, p)
Assumes a normal distribution for the object and an exponential distribution for occlusions.
Given the expected depth `μ` and object association probability `o`.
"""
DepthNormalUniform(μ, o, p::DepthImageParameters) = BinaryMixture(DepthNormal(μ, p), DepthUniform(p), o, 1.0 - o)

"""
  DepthNormalExponentialUniform(μ, o, p)
Assumes a normal distribution for the object and a mixture of an exponential and uniform distribution for occlusions and outliers.
Given the expected depth `μ` and object association probability `o`.
"""
DepthNormalExponentialUniform(μ, o, p::DepthImageParameters) = BinaryMixture(DepthNormal(μ, p), DepthExponentialUniform(p), o, 1.0 - o)

"""
  WrappedModel
Wrapper around an `AbstractMeasure`` for compatibility with AbstractMCMC's sample method.
"""
struct WrappedModel{T<:AbstractMeasure} <: AbstractMCMC.AbstractModel
  model::T
end

"""
  logdensity(pm, s)
Evaluates the logdensity of the internal model
"""
MeasureTheory.logdensity(pm::WrappedModel, s::Sample) = logdensity(pm.model, s)

"""
  rand(rng, pm)
Calls rand on the internal model
"""
Base.rand(rng::AbstractRNG, pm::WrappedModel) = rand(rng, pm.model)

"""
  prior_depth_model(model)
Model containing all the variables required to sample z.
"""
prior_depth_model(model) = prior(Model(model), :z)(argvals(model))

"""
  DepthImageMeasure(μ, o, m_pix, filter_fn, prep_fn)
Optimized measure for handling observation of depth images.

During inference it takes care of missing values in the expected depth `μ` and only evaluates the logdensity for pixels with depth 0 < z < max_depth.
Invalid values of z are set to zero by convention.

Each pixel is assumed to be independent and the measurement can be described by the measure `p.pixel_measure(μ, o)`.
`o` is the object association probability.

For the generative model, the whole image is generated.
"""
struct DepthImageMeasure <: AbstractMeasure
  μ::Matrix{Float64}
  o::Matrix{Float64}
  params::DepthImageParameters
end
MeasureTheory.basemeasure(::DepthImageMeasure) = Lebesgue{ℝ}
TransformVariables.as(d::DepthImageMeasure) = as(Array, asℝ, (d.params.width, d.params.height))

Base.show(io::IO, d::DepthImageMeasure) = print(io, "DepthImageMeasure\n  Parameters: $(d.params)")

# Generate independent random numbers from m_pix(μ, o)
function Base.rand(rng::AbstractRNG, T::Type, d::DepthImageMeasure)
  map(zip(d.μ, d.o)) do (μ, o)
    random_depth = rand(rng, T, d.params.pixel_measure(μ, o, d.params))
    d.params.min_depth < random_depth < d.params.max_depth ? random_depth : zero(random_depth)
  end
end

function MeasureTheory.logdensity(d::DepthImageMeasure, z)
  # Only sum the logdensity for values for which filter_fn is true
  ind = findall(x -> d.params.min_depth < x < d.params.max_depth, d.μ)
  sum = 0.0
  for i in ind
    # Preprocess the measurement
    z_i = d.params.min_depth < z[i] < d.params.max_depth ? z[i] : zero(z[i])
    sum = sum + logdensity(d.params.pixel_measure(d.μ[i], d.o[i], d.params), z_i)
  end
  sum
end

"""
  pixel_association(μ, z, q, params)
Probability of the pixel being associated to the object.
Given an expected depth of `μ` and observation of `z` the posterior is calculated using Bayes Law with the prior `q`.
The distribution of observing the object is constructed via `d_is(μ)` and other observations are explained by `d_not(μ)`
"""
function pixel_association(μ::Real, z::Real, q::Real, params::DepthImageParameters)::Float64
  # If the rendered value is invalid, we do not know more than before => prior
  if μ <= 0.0
    return q
  end
  prior_likelihood = pdf(params.association_is(μ, params), z) * q
  # Marginalize Bernoulli distributed by summing out o
  marginal = prior_likelihood + pdf(params.association_not(params), z) * (1 - q)
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
  image_association(s, z, prior_o, image_params, render_fn)
Uses the render image state of the sample `s.μ` and then calls `pixel_association` with each rendered pixel for the observed depth `z`.
"""
function image_association(s::Sample, z::AbstractMatrix{<:Real}, prior_o::AbstractMatrix{<:Real}, image_params::DepthImageParameters, render_fn::Base.Callable)
  st = state(s)
  μ = render_fn(st.t, st.r)
  # Also broadcast over the prior, protect params from broadcasting
  # TODO use previous sample as prior for o instead of prior_o?
  o = pixel_association.(μ, z, prior_o, (image_params,))
  tr = as((; o = as(Array, as𝕀, size(o))))
  Sample((; o = o), -Inf, tr)
end

"""
  nonzero_indices(img)
Returns a list of indices for the nonzero pixels in the image.
"""
nonzero_indices(img) = findall(!iszero, img)

