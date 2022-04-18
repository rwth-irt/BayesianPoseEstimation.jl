# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using MeasureTheory
using Random

# WARN MeasureTheory measures are not strongly typed, but KernelMeasures are. This means that rand(rng, MeasureTheory::AbstractMeasure) results in the default of Float64 computations.

"""
    ModelVariable(rng, measure)
Create a model variable by sampling from a measure.
"""
function ModelVariable(rng::AbstractRNG, measure::AbstractMeasure)
    val = rand(rng, measure)
    tr = as(measure)
    ModelVariable(val, tr)
end

ModelVariable(T::Type, measure::AbstractMeasure) = ModelVariable(Random.GLOBAL_RNG, T, measure)
ModelVariable(measure::AbstractMeasure) = ModelVariable(Random.GLOBAL_RNG, Float32, measure)


"""
    SampleVariable(rng, measure)
Create a sample variable by sampling from a measure.
"""
SampleVariable(rng::AbstractRNG, measure::AbstractMeasure) = ModelVariable(rng, measure) |> SampleVariable

SampleVariable(T::Type, measure::AbstractMeasure) = SampleVariable(Random.GLOBAL_RNG, T, measure)
SampleVariable(measure::AbstractMeasure) = SampleVariable(Random.GLOBAL_RNG, Float32, measure)

"""
    logdensity_var(measure, var)
Evaluate the uncorrected logdensity in the model domain for the variable.
"""
logdensity_var(measure::AbstractMeasure, var::AbstractVariable) = logdensity(measure, model_value(var))

"""
    logdensity_var()
Evaluate the uncorrected logdensity in the model domain for the variable.
For simple Measures, the conditional variables are neglected.
More involved models can be dispatched with their type.
"""
logdensity_var(measure::AbstractMeasure, var::AbstractVariable, ::NamedTuple{<:AbstractVariable}) = logdensity_var(measure, var)

"""
    logdensity_from_tuple(measure, value_logjac_tuple)
Serves as a kind of function barrier to make transform_logdensity type stable for CUDA.
"""
function logdensity_from_tuple(measure::AbstractMeasure, value_logjac_tuple::NTuple{2,<:Real})
    value, logjac = value_logjac_tuple
    logdensity(measure, value) + logjac
end


"""
    model_value_and_logjac(tr, raw_value)
A wrapper around transform_and_logjac which serves as a function barrier for type stability in CUDA.
Returns two arrays `(model_values, logjacs)` instead of one array of tuples.
"""
function model_value_and_logjac(transformation::TV.ScalarTransform, raw_values::AbstractArray)
    tuple = TV.transform_and_logjac.((transformation,), raw_values)
    model_values = first.(tuple)
    logjacs = last.(tuple)
    model_values, logjacs
end

"""
    model_value_and_logjac(tr, raw_value)
A wrapper around transform_and_logjac which allows us to only broadcast when necessary.
"""
model_value_and_logjac(tr::TV.ScalarTransform, raw_value::Real) = TV.transform_and_logjac(tr, raw_value)

"""
    accumulate_log_density(measure, log_densities, log_jacs)
Support of different accumulations for different measures.
General implementation is to sum up all the log_densities and log_jacs.
"""
accumulate_log_density(::AbstractMeasure, log_densities, log_jacs) = sum(log_densities) + sum(log_jacs)

"""
    accumulate_log_density(measure, log_densities, log_jacs)
Specialization for vectorized measures: accumulates the log_jacs to the last dimension of the measure size and adds them component wise to the log_densities.
"""
accumulate_log_density(d::VectorizedMeasure, log_densities, log_jacs) = log_densities .+ reduce_vectorized(+, d, log_jacs)

"""
    logdensity(measure, var)
Evaluate the logjac corrected logdensity in the model domain.
"""
function logdensity_var(measure::AbstractMeasure, var::SampleVariable)
    value, logjac = model_value_and_logjac(transformation(var), raw_value(var))
    log_density = logdensity(measure, value)
    accumulate_log_density(measure, log_density, logjac)
end

"""
    logdensity(measure, var)
Evaluate the logjac corrected logdensity in the model domain.
For simple Measures, the conditional variables are neglected.
More involved models can be dispatched with their type.
"""
logdensity_var(measure::AbstractMeasure, var::SampleVariable, ::NamedTuple) = logdensity_var(measure, var)
