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
    ModelVariable(rng, d)
Create a model variable by sampling from a kernel distribution.
"""
ModelVariable(rng::AbstractRNG, d::AbstractVectorizedDistribution, dims...) = ModelVariable(rand(rng, d, dims...), as(d))
ModelVariable(d::AbstractVectorizedDistribution, dims...) = ModelVariable(Random.GLOBAL_RNG, d, dims...)

"""
    SampleVariable(rng, d)
Create a sample variable by sampling from a kernel distribution.
"""
SampleVariable(rng::AbstractRNG, d::AbstractVectorizedDistribution, dims...) = ModelVariable(rng, d, dims...) |> SampleVariable
SampleVariable(d::AbstractVectorizedDistribution, dims...) = SampleVariable(Random.GLOBAL_RNG, d, dims...)

# TODO Distributions do not encode any information about variable names, thus logdensityof(measure, x::AbstractVariable, s::Sample) should not be required. Thus, encode it in the ModelInterface: logdensityof(::Model, ::Sample)

"""
    logdensity(d, c)
Evaluate the logjac corrected logdensity of the variable in the model domain.
"""
function DensityInterface.logdensityof(d::AbstractVectorizedDistribution, x::AbstractVariable)
    model_values, logjacs = model_value_and_logjac(x)
    # logdensityof vectorized distributions uses different reduction strategies
    # TODO this is copy pasted, can I create a better interface which allows me to inject the logjacs?
    log_densities = logdensityof(marginals(d), model_values)
    reduce_vectorized(+, d, logjacs + log_densities)
end


# TODO reduction should be moved to VectorizedDistributionsVariables.jl
# sum(log_density) + sum(logjac)

# # TODO move to KernelMeasureAdapter?
# """
#     accumulate_log_density(measure, log_densities, log_jacs)
# Specialization for vectorized measures: accumulates the log_jacs to the last dimension of the measure size and adds them component wise to the log_densities.
# """
# accumulate_log_density(d::KernelVectorized, log_densities, log_jacs) = log_densities .+ reduce_vectorized(+, d, log_jacs)
