# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using AbstractMCMC
using MeasureTheory
using Random

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
