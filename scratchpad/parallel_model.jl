# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

using AbstractMCMC
using Soss, MeasureTheory
using MCMCDepth

pixel_model = @model μ, σ, λ begin
  o ~ Uniform()
  z ~ BinaryMixture(Normal(μ, σ), Exponential(λ), o, 1.0 - o)
end

pix_params = pixel_model(σ=.1, λ=.5, o=0.4)

var_name = :μ
likelihood(x, z) = logdensity(pix_params((;var_name => x)) | (;z = z))

logdensity(pix_params((;var_name => 6.6)) | (;z=4.2))

likelihood(0.1, 1.2)

function pixel_likelihood(model, sample, variable)
end

using CUDA

v = CUDA.rand(100)

test(x) = logdensity(Normal(1.0,1.0), x)

soss_test(y) = logdensity(simple_model((;x=4.2)), (;y=y))

soss_test.(v)


struct PixelModel
  μ::Float32
  o::Float32
end

# Model needs to be conditioned on measurement for AbstractMCMC
struct ConditionedModel{M,Z}
  model::M
  z::Z
end

Base.:|(pm::PixelModel, z::Real) = ConditionedModel(pm, z)
pm = PixelModel(0.4, 0.5)
pm | 2.0

# logdensity(cm::ConditionedModel) = logdensity(cm.model, cm.z)

# TODO idea: provide a factory which generates the logdensity function of a model for a single parameter.
# This model needs to be conditioned on the data and all the other parameters.
# For more flexibility, condition the Model (Measure) on the other parameters first.

"""
  likelihood(model_factory, z)
Returns a likelihood function given the `model_factory` and measurement `z` which generates an `AbstractMeasure` with only the one parameter.
In other words: Conditione the model on the measurement.
"""
likelihood(model_factory::AbstractMeasure, z) = x -> logdensity(conditioned_model(x), z)

# WARN _... consumes other arguments, helpfull for conditioning on NamedTuples of Samples
"""
  pixel_model_factory(;μ, o)
Generator function for the pixel Measure.
General case and not conditioned on any 
"""
pixel_model_factory(;μ, o, σ, λ, _...) = BinaryMixture(Normal(μ, σ), Exponential(λ), o, 1.0 - o)
nt=(;μ=0.0, o=0.5)
model(;nt...)

"""
  partial(fn; kwargs...)
Partial applications of the keyword arguments, can be used to condition the model until only one argument is left.
"""
function partial(fn::Function; kwargs...)
  (; x...) -> fn(; kwargs..., x...)
end

test_partial(;x, y, z) = x+y+z
partial_fn = partial(test_partial; x=1.0)
partial_fn(;y=2.0, z=3.0)
partialfn_2  = partial(test_partial; x=1.0, y=2.0)
partialfn_2(;z=3.0)
partial_fn3 = partial_fn | (;y=2.0)
partial_fn3(;z=3.0)

"""
  |(fn, nt)
Syntactic sugar for partial(fn; kwargs...)
"""
(|)(fn::Function, nt::NamedTuple) = partial(fn; nt...)

# model conditioned on the other parameters
o_model(μ) = o -> model(μ=μ, o=o, x=1.0)
om = o_model(1.0)
om(1.0)




ℓ = likelihood(om, 2.0)
ℓ(0.5)

function model_generator(kwargs...)
  μ = kwargs[:μ]
  o = kwargs[:o]
end




"""
  InferenceModel
Split model into prior and posterior, which can be beneficial for some algorithms to avoid recalculations.
We assume that vectorization support is given for each model.
"""
abstract type InferenceModel <: AbstractMCMC.AbstractModel end

function serial_ℓ(model, samples::Sample)
  res = Vector{Float64}(undef, length(samples))
  for (i, s) in enumerate(samples)
    # TODO assumes that single sample logdensity is implemented
    res[i] = logdensity(model, s)
  end
end

prior_ℓ(model::InferenceModel, samples::AbstractVector{<:Sample}) = serial_ℓ(, samples)

function likelihood_ℓ(model::InferenceModel)
  res = Vector{Float64}(undef, length(samples))
  for (i, s) in enumerate(samples)
    # TODO assumes that single sample logdensity is implemented
    res[i] = likelihood_ℓ(model, s)
  end
end

function logdensity(model::InferenceModel, samples::AbstractVector{<:Sample})
  res = Vector{Float64}(undef, length(samples))
  prior_logdensity(model) + pixel_logdensity(model)(x)
end

function generate_ℓ(model::InferenceModel, samples::AbstractVector{<:Sample})
  res = Vector{Float64}(undef, length(samples))
  for (i, s) in enumerate(samples)
    # TODO assumes that single sample logdensity is implemented
    res[i] = generate_ℓ(model, s)
  end
end

using CUDA, StructArrays
struct Test
  a::Float32
  b::Vector{Float32}
end

t = Test(10, rand(Float32, 1000))
sa = StructArray([t for _ in 1:100])

da = replace_storage(CuArray, sa)

d = StructArray(a = rand(100), b = [rand(100) for _ in 1:100])

# move to GPU
dd = replace_storage(CuArray, d)