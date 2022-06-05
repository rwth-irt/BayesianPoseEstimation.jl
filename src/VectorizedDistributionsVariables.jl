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
# TODO test
ModelVariable(rng::AbstractRNG, dist::AbstractVectorizedDistribution, ::Sample, dims...) = ModelVariable(rng, dist, dims...)


"""
    SampleVariable(rng, dist)
Create a sample variable by sampling from a kernel distribution.
"""
SampleVariable(rng::AbstractRNG, dist::AbstractVectorizedDistribution, dims...) = ModelVariable(rng, dist, dims...) |> SampleVariable
SampleVariable(d::AbstractVectorizedDistribution, dims...) = SampleVariable(Random.GLOBAL_RNG, d, dims...)
# TODO test
SampleVariable(rng::AbstractRNG, d::AbstractVectorizedDistribution, ::Sample, dims...) = SampleVariable(rng, d, dims...)

# TODO should be dispatched correctly in VectorizedDistributions.jl
# TODO test
# """
#     logdensity(d, c)
# Evaluate the logjac corrected logdensity of the variable in the model domain.
# """
# function DensityInterface.logdensityof(d::AbstractVectorizedDistribution, x::AbstractVariable)
#     model_values, logjacs = model_value_and_logjac(x)
#     # logdensityof vectorized distributions uses different reduction strategies
#     # TODO this is copy pasted, can I create a better interface which allows me to inject the logjacs?
#     # logjac corrected from KernelDistributionsVariables.jl
#     reduce_vectorized(+, d, logdensityof(marginals(d), x))
# end

# TODO where do I need this? Proposals?
# DensityInterface.logdensityof(d::AbstractVectorizedDistribution, x::AbstractVariable, ::Sample) = logdensityof(d, x)
