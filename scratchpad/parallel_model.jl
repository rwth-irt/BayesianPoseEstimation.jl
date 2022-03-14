# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

# using AbstractMCMC
# using CUDA
using Cthulhu
using MeasureTheory
# WARN Soss model does not work with CUDA
# using Soss
using MCMCDepth


# # Model needs to be conditioned on measurement for AbstractMCMC
# struct ConditionedModel{M,Z}
#   model::M
#   z::Z
# end
# # logdensity(cm::ConditionedModel) = logdensity(cm.model, cm.z)

# # TODO idea: provide a factory which generates the logdensity function of a model for a single parameter.

test_partial(a; x, y) = a + x + y
test_partial(1; x = 2, y = 3)
test_a = MCMCDepth.partial(test_partial; x = 2, y = 3)
test_a(1)
@code_warntype test_a(1)
test_x = MCMCDepth.partial(test_partial, 1; y = 3)
test_x(; x = 2)
@code_warntype test_x(; x = 2)
test_args = MCMCDepth.kwarg_to_arg(test_x, :x)
test_args(2)
@code_warntype test_args(2)

# WARN _... consumes other arguments, helpfull for conditioning on NamedTuples of Samples
"""
  pixel_model_factory(;μ, o)
Generator function for the pixel Measure.
General case and not conditioned on any 
"""
pixel_model_factory(; μ, o, σ, λ, _...) = BinaryMixture(Normal(μ, σ), Exponential(λ), o, 1.0 - o)
nt = (; μ = 0.0, o = 0.5, λ = 1.5, σ = 0.1)
@code_warntype pixel_model_factory(; nt...)

factory_with_params = MCMCDepth.partial(pixel_model_factory; λ = 1.5, σ = 0.1)
μ_factory = MCMCDepth.partial(factory_with_params; o = 0.1)
@code_warntype μ_factory(; μ = 0.5)
# TODO this breaks type stability
μ_unstable(μ) = μ_factory(; (; :μ => μ)...)
@code_warntype μ_unstable(0.0)
# TODO this not????
μ_stable = MCMCDepth.kwarg_to_arg(test_x, :μ)
@code_warntype μ_stable(42.0)

# TODO basically does the same as partial function application
# More transparent what is happening in nt
# Less flexible regarding regular args
struct ConditionedFunction{F<:Base.Callable,NT<:NamedTuple}
  f::F
  nt::NT
end
# Make it callable
(cf::ConditionedFunction)(; b...) = cf.f(; b..., cf.nt...)

condition_fn(fn::Function, nt::NamedTuple) = ConditionedFunction(fn, nt)
condition_fn(cf::ConditionedFunction, nt) = ConditionedFunction(cf.f, merge(cf.nt, nt))
condition_fn(fn; kwargs...) = condition_fn(fn, values(kwargs))

test_condition(; x, y, z) = x + y + z
cf = condition_fn(test_condition; x = 1.0, y = 2.0)
@code_warntype condition_fn(cf; z = 3.0)()


# """
#   InferenceModel
# Split model into prior and posterior, which can be beneficial for some algorithms to avoid recalculations.
# We assume that vectorization support is given for each model.
# """
# abstract type InferenceModel <: AbstractMCMC.AbstractModel end

# function serial_ℓ(model, samples::Sample)
#   res = Vector{Float64}(undef, length(samples))
#   for (i, s) in enumerate(samples)
#     # TODO assumes that single sample logdensity is implemented
#     res[i] = logdensity(model, s)
#   end
# end

# prior_ℓ(model::InferenceModel, samples::AbstractVector{<:Sample}) = serial_ℓ(, samples)

# function likelihood_ℓ(model::InferenceModel)
#   res = Vector{Float64}(undef, length(samples))
#   for (i, s) in enumerate(samples)
#     # TODO assumes that single sample logdensity is implemented
#     res[i] = likelihood_ℓ(model, s)
#   end
# end

# function logdensity(model::InferenceModel, samples::AbstractVector{<:Sample})
#   res = Vector{Float64}(undef, length(samples))
#   prior_logdensity(model) + pixel_logdensity(model)(x)
# end

# function generate_ℓ(model::InferenceModel, samples::AbstractVector{<:Sample})
#   res = Vector{Float64}(undef, length(samples))
#   for (i, s) in enumerate(samples)
#     # TODO assumes that single sample logdensity is implemented
#     res[i] = generate_ℓ(model, s)
#   end
# end

# using CUDA, StructArrays
# struct Test
#   a::Float32
#   b::Vector{Float32}
# end

# t = Test(10, rand(Float32, 1000))
# sa = StructArray([t for _ in 1:100])

# da = replace_storage(CuArray, sa)

# d = StructArray(a = rand(100), b = [rand(100) for _ in 1:100])

# # move to GPU
# dd = replace_storage(CuArray, d)