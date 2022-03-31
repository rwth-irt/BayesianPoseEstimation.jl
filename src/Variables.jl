# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using CUDA
using TransformVariables

"""
Implement model_value and raw_value for the constrained model domain and the unconstrained sampling domain.
"""
abstract type AbstractVariable end

"""
    SampleVariable(value, transformation)
A single variable of a sample, which can have any `value` type depending on the algorithm (.e.g. scalar, cpu or gpu array).
By convention, we sample `value` in the unconstrained domain ℝ which can be converted to the constrained domain (e.g ℝ₊) by the `transformation`.
This must be a scalar transformation which can be broadcasted over the whole array.
"""
struct SampleVariable{V,T<:TransformVariables.ScalarTransform} <: AbstractVariable
    value::V
    transformation::T
end
# TODO might be smart or really stupid
SampleVariable(x, tr::TransformVariables.ArrayTransform) = SampleVariable(x, tr.transformation)
SampleVariable(x::AbstractVariable) = SampleVariable(raw_value(x), transformation(x))
Base.convert(::Type{SampleVariable}, x::AbstractVariable) = SampleVariable(x)

"""
    state(s)
Value of the variable in the model domain.
"""
model_value(s::SampleVariable) = transform.(s.transformation, s.value)

"""
    raw_value(s)
Value of the variable in the unconstrained domain.
"""
raw_value(s::SampleVariable) = s.value

"""
    transform(s)
Transform of the variable.
"""
transformation(s::SampleVariable) = s.transformation

"""
    ModelVariable(value, transformation)
A single variable of a sample, which can have any `value` type depending on the algorithm (.e.g. scalar, cpu or gpu array).
Compared to the SampleVariable, the `value` is in the constrained model domain.
The `transformation` must be a scalar transform which can be broadcasted over the whole array.
"""
struct ModelVariable{V,T<:TransformVariables.ScalarTransform} <: AbstractVariable
    value::V
    transformation::T
end

ModelVariable(x, tr::TransformVariables.ArrayTransform) = ModelVariable(x, tr.transformation)
ModelVariable(x::AbstractVariable) = ModelVariable(model_value(x), transformation(x))
Base.convert(::Type{ModelVariable}, x::AbstractVariable) = ModelVariable(x)

"""
    state(s)
Value of the variable in the model domain.
"""
model_value(s::ModelVariable) = s.value

"""
    raw_value(s)
Value of the variable in the unconstrained domain.
"""
raw_value(s::ModelVariable) = inverse.(s.transformation, s.value)

# WARN Should only be called once during the initialization so its probably not too costly.
"""
    raw_value(s)
Value of the variable in the unconstrained domain.
Executed on CPU, since `@argcheck` in TransformVariables cannot be compiled using CUDA.jl.
"""
raw_value(s::ModelVariable{<:CuArray}) = inverse.(s.transformation, Array(s.value)) |> CuArray

"""
    transform(s)
Transform of the variable.
"""
transformation(s::ModelVariable) = s.transformation

"""
Default is sampling in the unconstrained domain.
"""
Base.promote_rule(::Type{<:ModelVariable}, ::Type{<:SampleVariable}) = SampleVariable

"""
    +(a, b)
Add the values of two variables, in case of mixed types, `ModelVariable` gets promoted to `SampleVariable`.
If the transformations differ, the new sample has the one of `a`.
Vectorized by default. 
"""
function Base.:+(a::AbstractVariable, b::AbstractVariable)
    pa, pb = Base._promote(a, b)
    @set pa.value = pa.value .+ pb.value
end

"""
    -(a, b)
Subtract the values of `b` from `a`, in case of mixed types, `ModelVariable` gets promoted to `SampleVariable`.
If the transformations differ, the new sample has the one of `a`.
Vectorized by default. 
"""
function Base.:-(a::AbstractVariable, b::AbstractVariable)
    pa, pb = Base._promote(a, b)
    @set pa.value = pa.value .- pb.value
end
