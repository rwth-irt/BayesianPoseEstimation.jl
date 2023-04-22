# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# TODO Does it make more sense to return (;t=t,r=r) and reuse it in samplers where the prior is sampled?

"""
    point_prior(gl_context, params, experiment, cpu_rng)
Returns a BayesNet for μ(t,r) for an approximately known position and unknown orientation.
"""
function point_prior(gl_context, params, experiment, cpu_rng)
    t = BroadcastedNode(:t, cpu_rng, KernelNormal, experiment.prior_t, params.σ_t)
    r = BroadcastedNode(:r, cpu_rng, QuaternionUniform, params.float_type)

    μ_fn = render_fn | (gl_context, experiment.scene)
    DeterministicNode(:μ, μ_fn, (; t=t, r=r))
end

"""
    simple_posterior(params, experiment, μ_node, dev_rng)
A simple posterior model which does not calculate the pixel association probability `o` but uses a flat prior via `params.o`.
The pixel tail distribution is a mixture of an exponential and uniform distribution and checks for invalid values of `μ` via `ValidPixel`.
Provide a prior for `t, r` and the expected depth `μ` via the `μ_node`.
"""
function simple_posterior(params, experiment, μ_node, dev_rng)
    # NOTE almost no performance gain over DeterministicNode?
    o = BroadcastedNode(:o, dev_rng, KernelDirac, params.prior_o)
    # ValidPixel diverges without normalization
    z_i = pixel_valid_mixture | (params.min_depth, params.max_depth, params.pixel_θ, params.pixel_σ)
    z = BroadcastedNode(:z, dev_rng, z_i, (; μ=μ_node, o=o))
    z_norm = ModifierNode(z, dev_rng, ImageLikelihoodNormalizer | params.normalization_constant)
    PosteriorModel(z_norm, (; z=experiment.depth_image))
end

"""
    association_posterior(params, experiment, μ_node, dev_rng)
A posterior model which does calculate the pixel association probability `o`.
The pixel tail distribution is a mixture of an exponential and uniform distribution and checks for invalid values of `μ` via `ValidPixel`.
Provide a prior for `t, r` and the expected depth `μ` via the `μ_node`.
"""
function association_posterior(params, experiment, μ_node, dev_rng)
    o_fn = pixel_association_fn(params)
    o = DeterministicNode(:o, μ -> o_fn.(μ, experiment.depth_image), (; μ=μ_node))
    # ValidPixel diverges without normalization
    z_i = pixel_valid_mixture | (params.min_depth, params.max_depth, params.pixel_θ, params.pixel_σ)
    z = BroadcastedNode(:z, dev_rng, z_i, (; μ=μ_node, o=o))
    z_norm = ModifierNode(z, dev_rng, ImageLikelihoodNormalizer | params.normalization_constant)
    PosteriorModel(z_norm, (; z=experiment.depth_image))
end

"""
    smooth_posterior(params, experiment, μ_node, dev_rng)
A posterior model which does calculate the pixel association probability `o`.
The pixel tail distribution is a mixture of smoothed exponential and uniform distribution and checks for invalid values of `μ` via `ValidPixel`.
Provide a prior for `t, r` and the expected depth `μ` via the `μ_node`.
"""
function smooth_posterior(params, experiment, μ_node, dev_rng)
    # NOTE Analytic pixel association is only a deterministic function and not a Gibbs sampler in the traditional sense. Gibbs sampler would call rand(q(o|t,r,μ)) and not fn(μ,z). Probably "collapsed Gibbs" is the correct expression for it.
    o_fn = smooth_association_fn(params)
    # condition on data via closure
    o = DeterministicNode(:o, μ -> o_fn.(μ, experiment.depth_image), (; μ=μ_node))
    # ValidPixel diverges without normalization
    pixel_model = smooth_valid_mixture | (params.min_depth, params.max_depth, params.pixel_θ, params.pixel_σ)
    z = BroadcastedNode(:z, dev_rng, pixel_model, (; μ=μ_node, o=o))
    z_norm = ModifierNode(z, dev_rng, ImageLikelihoodNormalizer | params.normalization_constant)
    PosteriorModel(z_norm, (; z=experiment.depth_image))
end

