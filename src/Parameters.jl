# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 
using LinearAlgebra

"""
  DepthImageParameters
pix_σ: Standard deviation of the sensor noise.
pix_θ: Expected value of the exponential distribution → Occlusion expected closer or further away.
max_depth: Range limit of the sensor
mix_exponential: Mixture coefficient of the exponential part of the mixture.
width, height: Dimensions of the image.
pixel_measure(μ, o): Measure to evaluate the independent pixel logdensity
association_is(μ): Measure for the logdensity when the pixel belongs to the object (`o=1`)
association_not(): Measure for the logdensity when the pixel does not belong to the object (`o=0`)
prior_o: Prior probability of a pixel belonging to the object.
"""
Base.@kwdef struct DepthImageParameters{T<:Base.Callable,U<:Base.Callable,V<:Base.Callable}
  width::Int64 = 100
  height::Int64 = 100
  pix_σ::Float64 = 0.1
  pix_θ::Float64 = 1.0
  min_depth::Float64 = 0.1
  max_depth::Float64 = 10.0
  mix_exponential::Float64 = 0.8
  pixel_measure::T = DepthNormalExponentialUniform
  association_is::U = DepthNormal
  association_not::V = DepthExponential
end

"""
  PriorParameters
# Arguments
- `<...>_t` describes the prior `MvNormal` estimate from the RFID sensor and the pixel association.
- `o` describes the prior of each pixel belonging to the object of interest.
"""
Base.@kwdef struct PriorParameters
  mean_t::Vector{Float64} = [0.0, 0.0, 2.0]
  σ_t::Vector{Float64} = [0.05, 0.05, 0.05]
  cov_t::Matrix{Float64} = Diagonal(σ_t)
  o::Matrix{Float64} = fill(0.2, 100, 100)
end

"""
  RandomWalkParameters
Parameters of the zero centered normal distributions of a random walk proposal.
"""
Base.@kwdef struct RandomWalkParameters
  σ_t::Vector{Float64} = [0.05, 0.05, 0.05]
  cov_t::Matrix{Float64} = Diagonal(σ_t)
  σ_r::Vector{Float64} = [0.05, 0.05, 0.05]
  cov_r::Matrix{Float64} = Diagonal(σ_r)
  σ_o::Float64 = 0.05
  width::Int64 = 100
  height::Int64 = 100
end
