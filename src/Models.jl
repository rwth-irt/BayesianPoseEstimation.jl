# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Distributions

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
DepthExponential(μ) = Exponential(PIX_θ)

"""
  DepthUniform(μ, o)
Uniform distribution intended for observing random outliers.
Given the expected depth `μ`.
"""
DepthUniform(μ) = Uniform(0, MAX_DEPTH)

"""
  DepthExponentialUniform(μ, o)
Mixture of exponential and uniform distribution intended for observing an occlusion or random outlier.
Given the expected depth `μ`.
"""
DepthExponentialUniform(μ) = MixtureModel([DepthExponential(μ), DepthUniform(μ)], [MIX_EXPONENTIAL, MIX_UNIFORM])

"""
  DepthNormalExponential(μ, o)
Assumes a normal distribution for the object and an uniform distribution for random outliers.
Given the expected depth `μ` and object association probability `o`.
"""
DepthNormalExponential(μ, o) = MixtureModel([DepthNormal(μ), DepthExponential(μ)], [o, 1.0 - o])

"""
  DepthNormalUniform(μ, o)
Assumes a normal distribution for the object and an exponential distribution for occlusions.
Given the expected depth `μ` and object association probability `o`.
"""
DepthNormalUniform(μ, o) = MixtureModel([DepthNormal(μ), DepthUniform(μ)], [o, 1.0 - o])

"""
  DepthNormalExponentialUniform(μ, o)
Assumes a normal distribution for the object and a mixture of an exponential and uniform distribution for occlusions and outliers.
Given the expected depth `μ` and object association probability `o`.
"""
DepthNormalExponentialUniform(μ, o) = MixtureModel([DepthNormal(μ), DepthExponentialUniform(μ)], [o, 1.0 - o])

"""
  pixel_association(μ, z, q, p_is, p_not)
Probability of the pixel being associated to the object.
Given an expected depth of `μ` and observation of `z` the posterior is calculated using Bayes Law with the prior `q`.
The distribution of observing the object is constructed via `d_is(μ)` and other observations are explained by `d_not(μ)`
"""
function pixel_association(μ, z, q, d_is, d_not)
  # If the rendered value is invalid, we do not know more than before => prior
  if μ <= 0.0
    return q
  end
  prior_likelihood = pdf(d_is(μ), z) * q
  # Marginalize Bernoulli distributed by summing out o
  marginal = prior_likelihood + pdf(d_not(μ), z) * (1 - q)
  # Normalized posterior
  prior_likelihood / marginal
end

"""
  nonzero_indices(img)
Returns a list of indices for the nonzero pixels in the image.
"""
nonzero_indices(img) = findall(!iszero, img)

"""
  preprocess(render, img)
Reduce the computational load by operating only on the `indices`.
Recommendation: `nonzero_indices` to extract the non zero indices of the rendered image.
"""
function preprocess(indices, img) where {N}
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
# TODO How to make it type stable for N images without wrapping them in a Vector?
function preprocess(μ, o, z)
  ind = nonzero_indices(μ)
  preprocess(ind, μ), preprocess(ind, o), preprocess(ind, z)
end
