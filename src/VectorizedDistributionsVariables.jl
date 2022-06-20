# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using DensityInterface

"""
    VectorizedDistributionsVariables
Use VectorizedDistributions.jl with Variables.jl.
Provides constructors for ModelVariable and SampleVariable as well as the DensityInterface specialization for these.
"""

"""
    ModelVariable(rng, dist)
Create a model variable by sampling from a kernel distribution.
"""
ModelVariable(rng::AbstractRNG, dist::AbstractVectorizedDistribution, dims...) = ModelVariable(rand(rng, dist, dims...), bijector(dist))
ModelVariable(dist::AbstractVectorizedDistribution, dims...) = ModelVariable(Random.GLOBAL_RNG, dist, dims...)

"""
    SampleVariable(rng, dist)
Create a sample variable by sampling from a kernel distribution.
"""
SampleVariable(rng::AbstractRNG, dist::AbstractVectorizedDistribution, dims...) = ModelVariable(rng, dist, dims...) |> SampleVariable
SampleVariable(d::AbstractVectorizedDistribution, dims...) = SampleVariable(Random.GLOBAL_RNG, d, dims...)

# logdensityof(...) is dispatched correctly in VerctorizedDistributions.jl