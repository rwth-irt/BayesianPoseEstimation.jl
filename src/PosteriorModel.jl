# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# TODO move usings
using Bijectors
using DensityInterface
using Random

"""
    PosteriorModel(bijectors, data, prior, likelihood)
Consists of the `prior` and `likelihood` model.
On construction, the bijectors of the prior are eagerly evaluated, so samples can frequently be generated in the unconstrained domain and evaluated in the model domain - logjac correction included.
AbstractMCMC expects a model to be conditioned on the data, so it is included here.
"""
struct PosteriorModel{B<:NamedTuple,D<:NamedTuple,P<:SequentializedGraph,L<:SequentializedGraph} <: AbstractMCMC.AbstractModel
    bijectors::B
    data::D
    prior::P
    likelihood::L
end

function PosteriorModel(node::AbstractNode, data::NamedTuple{data_names}) where {data_names}
    sequentialized = sequentialize(node)
    # Only sample variables which are not conditioned on data
    prior_model = Base.structdiff(sequentialized, data)
    # Eagerly evaluate any lazily broadcasted bijectors
    bijectors = prior_model |> bijector |> map_materialize
    # Data conditioned nodes form the likelihood and are not transformed for sampling
    likelihood_model = sequentialized[data_names]
    PosteriorModel(bijectors, data, prior_model, likelihood_model)
end

Bijectors.bijector(posterior::PosteriorModel) = posterior.bijectors

"""
    rand(rng, posterior, dims...)
Generate a new sample from the prior in the unconstrained domain ℝⁿ.
"""
function Base.rand(posterior::PosteriorModel, dims::Integer...)
    # Generating a sample from the likelihood does only make sense for debugging and would not conform the typical Random interface
    prior_sample = rand(posterior.prior, dims...) |> Sample
    to_unconstrained_domain(prior_sample, posterior.bijectors)
end

# DensityInterface
@inline DensityKind(::PosteriorModel) = HasDensity()
"""
    logdensityof(posterior, sample, [ϕ=1])
Takes care of transforming the sample according to the bijectors of the prior and adding the logjac correction.
Allows tempering of the likelihood via ϕ:  p(θ|z) ∝ p(z|θ)ᵠ p(θ)
"""
DensityInterface.logdensityof(posterior::PosteriorModel, sample, ϕ=1) = add_logdensity(prior_and_likelihood(posterior, sample, ϕ)...)

"""
    logdensityof(posterior, sample, [ϕ=1])
Takes care of transforming the sample according to the bijectors of the prior and adding the logjac correction.
Allows tempering of the likelihood via ϕ:  p(θ|z) ∝ p(z|θ)ᵠ p(θ)
Returns ℓ_prior, ℓ_likelihood
"""
function prior_and_likelihood(posterior::PosteriorModel, sample, ϕ=1)
    model_sample, logjac = to_model_domain(sample, posterior.bijectors)
    ℓ_prior = logdensityof(posterior.prior, variables(model_sample))
    ℓ_prior_logjac = add_logdensity(ℓ_prior, logjac)
    # Early stopping if only the prior needs to be evaluated
    if iszero(ϕ)
        return ℓ_prior_logjac, one.(ℓ_prior_logjac)
    end

    conditioned_sample = merge(model_sample, posterior.data)
    ℓ_likelihood = logdensityof(posterior.likelihood, variables(conditioned_sample))
    if ϕ != 1
        # Often p(θ|z) ∝ p(z|θ)¹p(θ) is wanted -> save matrix multiplications
        ℓ_likelihood .*= ϕ
    end
    ℓ_prior_logjac, ℓ_likelihood
end