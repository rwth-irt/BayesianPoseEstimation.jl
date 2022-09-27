# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using AbstractMCMC
using Distributions
using MCMCDepth
using Plots
using Random

"""
Minimal example to sample from a known distribution in a constrained domain using proposals which live outside this domain.
"""

# Prepare RNG & result plots
rng = Random.default_rng()
plotly()
function plot_result_z((xmin, xmax), chain, bijectors, target)
    z_values = map(chain) do sample
        s, _ = to_model_domain(sample, bijectors)
        variables(s).z
    end
    histogram(z_values, bins=xmin:0.5:xmax, normalize=true)
    plot!(xmin:0.1:xmax, Base.Fix1(pdf, target); linewidth=3)
end

# Probabilistic model: prior and target distribution same domain [0,∞) for bijector test
target = Gamma(3.0, 1.0)
prior = IndependentModel((; z=Exponential()))
model = IndependentModel((; z=Gamma(3.0, 1.0)))

# Requires adjustment of prior
sym_proposal = SymmetricProposal(IndependentModel((; z=Normal(0, 0.1))))
sym_mh = MetropolisHastings(prior, sym_proposal)
sym_chain = sample(rng, model, sym_mh, 20000; discard_initial=0, thinning=0);
plot_result_z((0, 15), sym_chain, bijector(prior), target)

# Requires adjustment of prior
ind_proposal = IndependentProposal(IndependentModel((; z=Normal())))
ind_mh = MetropolisHastings(prior, ind_proposal)
ind_chain = sample(rng, model, ind_mh, 50000; discard_initial=0, thinning=0);
plot_result_z((0, 15), ind_chain, bijector(prior), target)

# Requires adjustment of prior & proposal
ind_proposal = IndependentProposal(IndependentModel((; z=Uniform())))
ind_mh = MetropolisHastings(prior, ind_proposal)
ind_chain = sample(rng, model, ind_mh, 50000; discard_initial=0, thinning=0);
plot_result_z((0, 15), ind_chain, bijector(prior), target)

# Requires adjustment of prior & proposal
ind_proposal = IndependentProposal(IndependentModel((; z=Exponential())))
ind_mh = MetropolisHastings(prior, ind_proposal)
ind_chain = sample(rng, model, ind_mh, 50000; discard_initial=0, thinning=0);
plot_result_z((0, 15), ind_chain, bijector(prior), target)
