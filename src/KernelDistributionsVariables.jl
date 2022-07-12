# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Base: broadcasted
using DensityInterface
using Random

"""
    KernelDistributionsAdapter
Use KernelDistributions.jl with Variables.jl.
Provides constructors for `ModelVariable` and `SampleVariable` as well as the DensityInterface specialization for these.
"""

"""
    ModelVariable(rng, dist)
Create a `ModelVariable` by sampling from a `KernelDistribution`.
"""
ModelVariable(rng::AbstractRNG, dist::KernelOrKernelArray, dims...) = ModelVariable(rand(rng, dist, dims...), bijector(dist))
ModelVariable(dist::KernelOrKernelArray, dims...) = ModelVariable(Random.GLOBAL_RNG, dist, dims...)

"""
    logdensityof(dist, variable)
Evaluate the logdensity of the distribution `dist` and `variable` in the model domain.
ModelVariables can be evaluated without transformation and logjac correction.
"""
DensityInterface.logdensityof(dist::AbstractKernelDistribution, variable::ModelVariable) = logdensityof.(dist, model_value(variable))
DensityInterface.logdensityof(dist::AbstractArray{<:AbstractKernelDistribution}, variable::ModelVariable) = logdensityof.(dist, model_value(variable))

"""
    SampleVariable(rng, dist)
Create a `SampleVariable` by sampling from a `KernelDistribution`.
"""
SampleVariable(rng::AbstractRNG, dist::KernelOrKernelArray, dims...) = ModelVariable(rng, dist, dims...) |> SampleVariable
SampleVariable(dist::KernelOrKernelArray, dims...) = SampleVariable(Random.GLOBAL_RNG, dist, dims...)

"""
    logjac_corrected_logdensityof(dist, unconstrained_value, bijector)
Serves as function barrier for broadcasting so it is compiled as a kernel and avoids allocations.
"""
function logjac_corrected_logdensityof(dist, unconstrained_value, bijector::Bijector)
    # TODO logendsityof.(transformed.(d), raw_value(x)) would be elegant but would require that I implement Distributions.jl for KernelDistributions.jl or a custom TransformedDistribution.
    # This way it is explicit, like in my dissertation
    model_value, jac = with_logabsdet_jacobian(inverse(bijector), unconstrained_value)
    logdensityof(dist, model_value) + jac
end

# TODO Method ambiguities. For the DensityInterface it will probably make more sense to implement transformed distributions and store raw values in the Sample. Also see the above.
"""
    logdensityof(dist, variable)
Evaluate the logjac corrected logdensity of the distribution `dist` and `variable` in the model domain.
"""
DensityInterface.logdensityof(dist::AbstractKernelDistribution, variable::SampleVariable) = logjac_corrected_logdensityof.(dist, raw_value(variable), bijector(variable))

"""
    logdensityof(dists, variable)
Evaluate the logjac corrected logdensity of the distribution `dist` and `variable` in the model domain.
"""
function DensityInterface.logdensityof(dists::AbstractArray{<:AbstractKernelDistribution}, variable::SampleVariable)
    unconstrained_value = raw_value(variable)
    # TODO let it fail?
    device_dists = maybe_cuda(unconstrained_value, dists)
    logjac_corrected_logdensityof.(device_dists, unconstrained_value, bijector(variable))
end

"""
    logdensityof(dist, variable)
Evaluate the logjac corrected logdensity of the distribution `dist` and `variable` in the model domain.
Special case: scalar numbers would be returned as tuple when broadcasting.
"""
DensityInterface.logdensityof(dist::AbstractKernelDistribution, variable::SampleVariable{<:Number}) = logjac_corrected_logdensityof(dist, raw_value(variable), bijector(variable))
