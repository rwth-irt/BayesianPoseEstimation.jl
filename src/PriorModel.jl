# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 
# Samples

using Random
using TransformVariables

"""
    AbstractPriorModel
Implement Random.rand(rng, type, model) => Sample and MeasureTheory.logdensity(model, sample) => Real
"""
abstract type AbstractPriorModel end
Random.rand(T::Type, m::AbstractPriorModel) = rand(Random.GLOBAL_RNG, T, m)
Random.rand(m::AbstractPriorModel) = rand(Random.GLOBAL_RNG, Float32, m)

"""
    IndependentPrior
Model variables are independent so the sampling order does not matter
"""
struct IndependentPrior{K,V} <: AbstractPriorModel
    models::NamedTuple{K,V}
end

"""
    rand(rng, T, prior)
Create a random sample  from the IndependentPrior.
"""
function Random.rand(rng::AbstractRNG, T::Type, prior::IndependentPrior{K}) where K
    vars = map(prior.models) do m
        SampleVariable(rng, T, m)
    end
    Sample(NamedTuple{K}(vars), -Inf)
end

"""
    logdensity(models, sample)
Evaluate the logdensity in the model domain.
Logjac correction is automatically applied to SampleVariables but not to any other AbstractVariable
"""
function MeasureTheory.logdensity(prior::IndependentPrior, sample::Sample)
    v = vars(sample)
    ℓ = map_models(logdensity_var, prior.models, v)
    # TODO CUDA likelihoods should be copied to array in the internal logdensity
    reduce(.+, ℓ)
end

# WARN logpdf is not typestable due to MeasureTheory internals. Probably not used anyways. so not implemented

"""
    PoseDepthPrior
Consists of an independent prior for the translation `t`, rotation `r`, and object association `o` and a function to render the expected depth `μ`.
"""
struct PoseDepthPrior{V}
    tro_model::IndependentPrior{(:t, :r, :o),V}
    render_fn::Function
    # TODO use tiles here?
end

PoseDepthPrior(r_model, t_model, o_model, render_fn) = PoseDepthPrior(IndependentPrior(; r=r_model, t=t_model, o=o_model), render_fn)

function render_sample()

function Random.rand(rng::AbstractRNG, T::Type, prior::PoseDepthPrior)
    tro_sample = rand(rng, T, prior.tro_model)
    t = model_value(tro_sample, :t)
    r = model_value(tro_sample, :o)
    # TODO multiple hypotheses by implementing rand([rng=GLOBAL_RNG], [S], [dims...]) ? But us tiles instead of dims? Or dispatch on VectorizedMeasures - would still need Tiles. Probably best of both worlds: Render number on basis of vectorized measures in the tiled texture. For uniform interface it is probably best to include tiles in all Depth models. DepthModels.jl with AbstractDepthModel? tiles(::AbstractDepthModel)
    μ = prior.render_fn(t, r)
    μ_var = ModelVariable(μ, asℝ)
    μ_sample = Sample((;μ=μ_var), -Inf)
    merge(tro_sample, μ_sample)
end
