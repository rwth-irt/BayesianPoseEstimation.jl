# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# TODO move usings
using Bijectors
using DensityInterface
using Random

"""
    PosteriorModel(data, prior, likelihood)
Implement the following for `prior` and `likelihood`:
- `bijector(prior)`
- `rand(rng, prior, sample, dims...)`
- `logdensityof(prior, sample)`
- `logdensityof(likelihood, sample)`

On construction, the bijectors of the prior are eagerly evaluated, so samples can frequently be generated in the unconstrained domain and evaluated in the model domain - logjac correction included.
AbstractMCMC expects a model to be conditioned on the data, so it is included here.
"""
struct PosteriorModel{B<:NamedTuple{<:Any,<:Tuple{Vararg{<:Bijector}}},P,L<:ConditionedModel} <: AbstractMCMC.AbstractModel
    bijectors::B
    prior::P
    likelihood::L
end

function PosteriorModel(data::NamedTuple, prior, likelihood)
    bijectors = map_materialize(bijector(prior))
    cond_likelihood = ConditionedModel(data, likelihood)
    PosteriorModel(bijectors, prior, cond_likelihood)
end

"""
    rand(rng, posterior, dims...)
Generate a new sample from the prior in the unconstrained domain ℝⁿ.
"""
function Base.rand(rng::AbstractRNG, posterior::PosteriorModel, dims::Integer...)
    prior_sample = rand(rng, posterior.prior, dims...)
    to_unconstrained_domain(prior_sample, posterior.bijectors)
    # Generating a sample from the likelihood does only make sense for debugging and would not conform the typical Random interface
end

# DensityInterface
@inline DensityKind(::PosteriorModel) = HasDensity()
"""
    logdensityof(posterior, sample)
Takes care of transforming the sample according to the bijectors of the prior and adding the logjac correction.
"""
function DensityInterface.logdensityof(posterior::PosteriorModel, sample)
    model_sample, logjac = to_model_domain(sample, posterior.bijectors)
    ℓ_prior = logdensityof(posterior.prior, model_sample)
    ℓ_likelihood = logdensityof(posterior.likelihood, model_sample)
    .+(promote(ℓ_prior, ℓ_likelihood, logjac)...)
end

prior(posterior::PosteriorModel) = posterior.prior
Bijectors.bijector(posterior::PosteriorModel) = posterior.bijectors
