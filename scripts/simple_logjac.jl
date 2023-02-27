# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

using AbstractMCMC
using KernelDistributions
using MCMCDepth
using Plots
using Random

"""
Minimal example to sample from a known distribution in a constrained domain using proposals which live outside this domain.
"""

# Prepare RNG & result plots
rng = Random.default_rng()
gr()
function plot_result_z((xmin, xmax), chain, bijectors, target)
    z_values = map(chain) do sample
        s, _ = to_model_domain(sample, bijectors)
        variables(s).z
    end
    histogram(z_values, bins=xmin:0.5:xmax, normalize=true)
    plot!(xmin:0.1:xmax, Base.Fix1(pdf, target); linewidth=3)
end

# Probabilistic model: domain [0,∞) for bijector test
target = KernelExponential(3.0f0)
model = SimpleNode(:z, rng, KernelExponential, 3.0f0)
posterior = PosteriorModel(model, (;))

# Requires adjustment of prior
sym_proposal = symmetric_proposal(SimpleNode(:z, rng, KernelNormal, 0, 0.1), model)
sym_mh = MetropolisHastings(sym_proposal)
sym_chain = sample(rng, posterior, sym_mh, 10_000; discard_initial=0, thinning=5);
plot_result_z((0, 15), sym_chain, bijector(posterior), target)

# Requires adjustment of prior
ind_proposal = independent_proposal(SimpleNode(:z, rng, KernelNormal), model)
ind_mh = MetropolisHastings(ind_proposal)
ind_chain = sample(rng, posterior, ind_mh, 10_000; discard_initial=0, thinning=5);
plot_result_z((0, 15), ind_chain, bijector(posterior), target)

# Requires adjustment of prior & proposal
ind_proposal = independent_proposal(SimpleNode(:z, rng, KernelUniform), model)
ind_mh = MetropolisHastings(ind_proposal)
ind_chain = sample(rng, posterior, ind_mh, 10_000; discard_initial=0, thinning=5);
plot_result_z((0, 15), ind_chain, bijector(posterior), target)

# Requires adjustment of prior & proposal
ind_proposal = independent_proposal(SimpleNode(:z, rng, KernelExponential), model)
ind_mh = MetropolisHastings(ind_proposal)
ind_chain = sample(rng, posterior, ind_mh, 10_000; discard_initial=0, thinning=5);
plot_result_z((0, 15), ind_chain, bijector(posterior), target)

# Test whether the combination of the two samplers works
com_sampler = ComposedSampler(sym_mh, ind_mh)
ind_chain = sample(rng, posterior, com_sampler, 10_000; discard_initial=0, thinning=5);
plot_result_z((0, 15), ind_chain, bijector(posterior), target)