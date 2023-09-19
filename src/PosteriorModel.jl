# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# TODO move to BayesNet or is the bijectors stuff not agnostic enough?

"""
    PosteriorModel(bijectors, prior, likelihood)
Consists of the `prior` and `likelihood` model.
On construction, the bijectors of the prior are eagerly evaluated, so samples can frequently be generated in the unconstrained domain and evaluated in the model domain - logjac correction included.
AbstractMCMC expects a model to be conditioned on the data, so it is included here.
"""
struct PosteriorModel{B<:NamedTuple,P<:SequentializedGraph,L<:SequentializedGraph} <: AbstractMCMC.AbstractModel
    bijectors::B
    prior::P
    likelihood::L
end

function PosteriorModel(graph::SequentializedGraph)
    obs_names = findall(x -> isa(x, ObservationNode), graph)
    obs_nodes = isempty(obs_names) ? (;) : graph[obs_names]
    # Only sample variables which are not conditioned on data
    prior_nodes = Base.structdiff(graph, obs_nodes)
    # Eagerly evaluate any lazily broadcasted bijectors
    bijectors = prior_nodes |> bijector |> map_materialize
    PosteriorModel(bijectors, prior_nodes, obs_nodes)
end

# TODO replace with BayesNet::ObservationNode s
PosteriorModel(root_node::AbstractNode) = PosteriorModel(sequentialize(root_node))

Base.show(io::IO, posterior::PosteriorModel) = print(io, "PosteriorModel(prior for $(keys(posterior.prior)), likelihood for $(keys(posterior.likelihood)) & bijectors for $(keys(posterior.bijectors)))")

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
    logdensityof(posterior, sample)
Takes care of transforming the sample according to the bijectors of the prior and adding the logjac correction.
"""
DensityInterface.logdensityof(posterior::PosteriorModel, sample) = add_logdensity(prior_and_likelihood(posterior, sample)...)

"""
    logdensityof(posterior, sample)
Takes care of transforming the sample according to the bijectors of the prior and adding the logjac correction.
Returns ℓ_prior, ℓ_likelihood
"""
function prior_and_likelihood(posterior::PosteriorModel, sample)
    model_sample, logjac = to_model_domain(sample, posterior.bijectors)
    ℓ_prior = logdensityof(posterior.prior, variables(model_sample))
    ℓ_prior_logjac = add_logdensity(ℓ_prior, logjac)
    ℓ_likelihood = logdensityof(posterior.likelihood, variables(model_sample))
    # Do not broadcast over tuple to avoid invocation of GPU compiler
    to_cpu(ℓ_prior_logjac), to_cpu(ℓ_likelihood)
end

function logdensity_sample(posterior::PosteriorModel, sample)
    ℓ_prior, ℓ_likelihood = prior_and_likelihood(posterior, sample)
    ℓ_posterior = add_logdensity(ℓ_prior, ℓ_likelihood)
    Sample(variables(sample), ℓ_posterior, ℓ_likelihood)
end

function tempered_logdensity_sample(posterior::PosteriorModel, sample, temp)
    ℓ_prior, ℓ_likelihood = prior_and_likelihood(posterior, sample)
    ℓ_posterior = tempered_logdensity(ℓ_prior, ℓ_likelihood, temp)
    Sample(variables(sample), ℓ_posterior, ℓ_likelihood)
end

function tempered_logdensity(log_prior, log_likelihood, temp=1)
    if temp == 0
        return log_prior
    end
    if temp == 1
        return add_logdensity(log_prior, log_likelihood)
    end
    add_logdensity(log_prior, temp .* log_likelihood)
end
