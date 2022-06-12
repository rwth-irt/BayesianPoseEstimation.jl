# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using Bijectors
using CUDA
using Logging
# using SciGL

"""
Implement model_value and raw_value for the constrained model domain and the unconstrained sampling domain.
"""
abstract type AbstractVariable end

"""
    value(variable)
Returns the internal value which might be in the model or the unconstrained domain.
Only for internal use, e.g. zero(value(variable))
"""
value(variable::AbstractVariable) = variable.value

"""
    SampleVariable(value, bijector)
A single variable of a sample, which can have any `value` type depending on the algorithm (.e.g. scalar, cpu or gpu array).
By convention, we sample `value` in the unconstrained domain ℝ which can be converted to the constrained domain (e.g ℝ₊) by the `bijector`.
"""
struct SampleVariable{V,B<:Bijector} <: AbstractVariable
    value::V
    bijector::B
end
SampleVariable(var::AbstractVariable) = SampleVariable(raw_value(var), bijector(var))
Base.convert(::Type{SampleVariable}, var::AbstractVariable) = SampleVariable(var)

"""
    model_value(var)
Value of the variable in the model domain.
"""
model_value(var::SampleVariable) = inverse(var.bijector)(var.value)

"""
    model_value_with_logjac(var)
Tuple of the value in the model domain and the logabsdet_jacobian correction.
"""
Bijectors.with_logabsdet_jacobian(var::SampleVariable) = with_logabsdet_jacobian(inverse(var.bijector), var.value)

"""
    raw_value(var)
Value of the variable in the unconstrained domain.
"""
raw_value(var::SampleVariable) = var.value

"""
    bijector(var)
Bijector of the variable.
"""
Bijectors.bijector(var::SampleVariable) = var.bijector

"""
    ModelVariable(value, bijector)
A single variable of a sample, which can have any `value` type depending on the algorithm (.e.g. scalar, cpu or gpu array).
Compared to the SampleVariable, the `value` is in the constrained model domain.
"""
struct ModelVariable{V,B<:Bijector} <: AbstractVariable
    value::V
    bijector::B
end
ModelVariable(var::AbstractVariable) = ModelVariable(model_value(var), bijector(var))
Base.convert(::Type{ModelVariable}, var::AbstractVariable) = ModelVariable(var)

"""
    model_value(var)
Value of the variable in the model domain.
"""
model_value(var::ModelVariable) = var.value

"""
    model_value_with_logjac(var)
Tuple of the value in the model domain and the logabsdet_jacobian correction.
"""
Bijectors.with_logabsdet_jacobian(var::ModelVariable) = var.value, zero(var.value)

"""
    raw_value(var)
Value of the variable in the unconstrained domain.
"""
raw_value(var::ModelVariable) = bijector(var)(var.value)

"""
    bijector(var)
Bijector of the variable.
"""
Bijectors.bijector(var::ModelVariable) = var.bijector

"""
Default is sampling in the unconstrained domain.
"""
Base.promote_rule(::Type{<:ModelVariable}, ::Type{<:SampleVariable}) = SampleVariable

"""
    +(var_a, var_b)
Add the values of two variables, in case of mixed types, `ModelVariable` gets promoted to `SampleVariable`.
If the bijectors differ, the new sample has the one of `var_a`.
Vectorized by default. 
"""
function Base.:+(var_a::AbstractVariable, var_b::AbstractVariable)
    pa, pb = Base._promote(var_a, var_b)
    @set pa.value = pa.value .+ pb.value
end

"""
    -(var_a, var_b)
Subtract the values of `var_b` from `var_a`, in case of mixed types, `ModelVariable` gets promoted to `SampleVariable`.
If the bijectors differ, the new sample has the one of `var_a`.
Vectorized by default. 
"""
function Base.:-(var_a::AbstractVariable, var_b::AbstractVariable)
    pa, pb = Base._promote(var_a, var_b)
    @set pa.value = pa.value .- pb.value
end
