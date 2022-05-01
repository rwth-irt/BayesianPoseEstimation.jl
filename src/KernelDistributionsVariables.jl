# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using DensityInterface

"""
    KernelDistributionsAdapter
Use KernelDistributions.jl with Variables.jl.
Provides constructors for ModelVariable and SampleVariable as well as the DensityInterface specialization for these.
"""

"""
    ModelVariable(rng, d)
Create a model variable by sampling from a vectorized distribution.
"""
ModelVariable(rng::AbstractRNG, d::KernelOrKernelArray, dims...) = ModelVariable(rand(rng, d, dims...), as(d))
ModelVariable(d::KernelOrKernelArray, dims...) = ModelVariable(Random.GLOBAL_RNG, d, dims...)

"""
    SampleVariable(rng, d)
Create a sample variable by sampling from a kernel distribution.
"""
SampleVariable(rng::AbstractRNG, d::KernelOrKernelArray, dims...) = ModelVariable(rng, d, dims...) |> SampleVariable
SampleVariable(d::KernelOrKernelArray, dims...) = SampleVariable(Random.GLOBAL_RNG, d, dims...)

"""
    logdensity(d, c)
Evaluate the logjac corrected logdensity of the variable in the model domain.
"""
function DensityInterface.logdensityof(d::KernelOrKernelArray, x::AbstractVariable)
    model_value, logjac = model_value_and_logjac(x)
    log_density = logdensityof(d, model_value)
    logjac + log_density
end


