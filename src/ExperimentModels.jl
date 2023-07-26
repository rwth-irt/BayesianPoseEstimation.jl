# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# TODO Include prior_o in functions - assume either RFID or mask to be available - finally both?

"""
    point_prior(params, experiment, cpu_rng)
Returns a BayesNet for μ(t,r) for an approximately known position and unknown orientation.
"""
function point_prior(params::Parameters, experiment::Experiment, cpu_rng::AbstractRNG)
    t = BroadcastedNode(:t, cpu_rng, KernelNormal, experiment.prior_t, params.σ_t)
    r = BroadcastedNode(:r, cpu_rng, QuaternionUniform, params.float_type)

    μ_fn = render_fn | (experiment.gl_context, experiment.scene)
    DeterministicNode(:μ, μ_fn, (t, r))
end

# TODO move arguments to Experiment
# TODO add to Diss
function segmentation_prior(params::Parameters, experiment::Experiment, cpu_rng::AbstractRNG, mask_img, bbox, cv_camera)
    # u & v are the center of the bounding box
    left, right, top, bottom = bbox
    u, v = (left + right, top + bottom) ./ 2
    # Assumption: Most pixels belong to the object.
    masked = experiment.depth_image[mask_img.>0]
    mean_z = mean(masked)
    reproject_3D(u, v, mean_z, cv_camera)

    # When dividing masked by some amount, the mean also changes the same way
    # NOTE scaling is an arbitrary Hyperparameter -> parameters
    scaling = 0.1
    σ_z = std(masked .* scaling, mean=mean_z * scaling)
    σ_t = copy(params.σ_t)
    σ_t[3] = σ_z
    @reset params.σ_t = σ_z

    point_prior(params, experiment, cpu_rng)
end

"""
    simple_posterior(params, experiment, μ_node, dev_rng)
A simple posterior model which does not calculate the pixel association probability `o` but uses a fixed prior via `params.o`.
The pixel tail distribution is a mixture of an exponential and uniform distribution and checks for invalid values of `μ` via `ValidPixel`.
Provide a prior for `t, r` and the expected depth `μ` via the `μ_node`.
"""
function simple_posterior(params, experiment, μ_node, dev_rng)
    o = BroadcastedNode(:o, dev_rng, KernelDirac, experiment.prior_o)
    # ValidPixel diverges without normalization
    z_i = pixel_valid_mixture | (params.min_depth, params.max_depth, params.pixel_θ, params.pixel_σ)
    z = BroadcastedNode(:z, dev_rng, z_i, (μ_node, o))
    # NOTE seems to work better if mask is available
    # z_norm = ModifierNode(z, dev_rng, ImageLikelihoodNormalizer | params.normalization_constant)
    z_norm = ModifierNode(z, dev_rng, SimpleImageRegularization)
    PosteriorModel(z_norm | experiment.depth_image)
end

"""
    association_posterior(params, experiment, μ_node, dev_rng)
A posterior model which does calculate the pixel association probability `o`.
The pixel tail distribution is a mixture of an exponential and uniform distribution and checks for invalid values of `μ` via `ValidPixel`.
Provide a prior for `t, r` and the expected depth `μ` via the `μ_node`.
"""
function association_posterior(params, experiment, μ_node, dev_rng)
    o_fn = pixel_association_fn(params)
    # condition on data via closure
    o = DeterministicNode(:o, μ -> o_fn.(experiment.prior_o, μ, experiment.depth_image), (μ_node,))
    # ValidPixel diverges without normalization
    z_i = pixel_valid_mixture | (params.min_depth, params.max_depth, params.pixel_θ, params.pixel_σ)
    z = BroadcastedNode(:z, dev_rng, z_i, (μ_node, o))
    # NOTE seems to perform better with ImageLikelihoodNormalizer if prior is known for o. Also seems to perform worse than the simple_posterior if SimpleImageRegularization is used.
    z_norm = ModifierNode(z, dev_rng, ImageLikelihoodNormalizer | params.normalization_constant)
    PosteriorModel(z_norm | experiment.depth_image)
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
    o = DeterministicNode(:o, μ -> o_fn.(experiment.prior_o, μ, experiment.depth_image), (μ_node,))
    # ValidPixel diverges without normalization
    pixel_model = smooth_valid_mixture | (params.min_depth, params.max_depth, params.pixel_θ, params.pixel_σ)
    z = BroadcastedNode(:z, dev_rng, pixel_model, (μ_node, o))
    z_norm = ModifierNode(z, dev_rng, ImageLikelihoodNormalizer | params.normalization_constant)
    PosteriorModel(z_norm | experiment.depth_image)
end
