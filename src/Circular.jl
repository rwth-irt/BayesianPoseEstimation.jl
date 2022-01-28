# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Bijectors
using Random
using Turing

"""
  Circular
Bijection (0,2π) → ℝ which handles the periodicity by using `mod2pi`.
"""
struct Circular <: Bijector{0} end

(::Circular)(x) = x
(::Inverse{<:Circular})(y) = mod2pi(y)
Bijectors.logabsdetjac(::Circular, y) = zero(y)

"""
  CircularUniform
Uniform distribution ∈ (0,2π) which handles the periodicity using the Circular bijector.
"""
struct CircularUniform <: ContinuousUnivariateDistribution end
Distributions.rand(rng::AbstractRNG, ::CircularUniform) = rand(rng, Uniform(0, 2π))
Distributions.logpdf(::CircularUniform, ::Real) = log(1 / 2π)

Bijectors.bijector(::CircularUniform) = Circular()