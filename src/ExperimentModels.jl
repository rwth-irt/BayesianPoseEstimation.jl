# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

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

"""
    segmentation_to_point(bounding_box, depth_image, mask_img, cv_camera)
Calculates a 3D point which can be used for `prior_t` in the `Experiment` with x and y at the center of the bounding box and z as the mean of the masked depth image.
The `bounding_box` should contain the mask at its center.  
"""
function point_from_segmentation(bounding_box, depth_image, mask_img, cv_camera)
    # u & v are the center of the bounding box
    left, right, top, bottom = bounding_box
    u, v = (left + right, top + bottom) ./ 2
    # Assumption: Most pixels belong to the object.
    masked = depth_image[mask_img.>=0]
    z = median(masked)
    x, y = reproject_3D(u, v, z, cv_camera)
    [x, y, z]
end

"""
    simple_posterior(params, experiment, μ_node, dev_rng)
A simple posterior model which does not calculate the pixel association probability `o` but uses a fixed prior via `params.o`.
The pixel tail distribution is a mixture of an exponential and uniform distribution.
Provide a prior for `t, r` and the expected depth `μ` via the `μ_node`.
"""
function simple_posterior(params, experiment, μ_node, dev_rng)
    o = BroadcastedNode(:o, dev_rng, KernelDirac, experiment.prior_o)
    # NOTE Not checking for μ > 0 works well with SimpleImageRegularization
    z_i = pixel_mixture | (params.min_depth, params.max_depth, params.pixel_θ, params.pixel_σ)
    z = BroadcastedNode(:z, dev_rng, z_i, (μ_node, o))
    # NOTE If o is not estimated, the ImageLikelihoodNormalizer boils down to SimpleImageRegularization as sum(o .>= 0.5) is constant
    z_norm = ModifierNode(z, dev_rng, SimpleImageRegularization | params.c_reg)
    PosteriorModel(z_norm | experiment.depth_image)
end

"""
    association_posterior(params, experiment, μ_node, dev_rng)
A posterior model which does calculate the pixel association probability `o`.
The pixel tail distribution is a mixture of an exponential and uniform distribution.
Provide a prior for `t, r` and the expected depth `μ` via the `μ_node`.
"""
function association_posterior(params, experiment, μ_node, dev_rng)
    o_fn = pixel_association_fn(params)
    # condition on data via closure
    o = DeterministicNode(:o, μ -> o_fn.(experiment.prior_o, μ, experiment.depth_image), (μ_node,))
    # NOTE ValidPixel diverges without regularization. Moreover, ValidPixel is not required if a regularization is used.
    z_i = pixel_mixture | (params.min_depth, params.max_depth, params.pixel_θ, params.pixel_σ)
    z = BroadcastedNode(:z, dev_rng, z_i, (μ_node, o))
    # NOTE seems to perform better with ImageLikelihoodNormalizer if prior is known for o. Also seems to perform worse than the simple_posterior if SimpleImageRegularization is used.
    z_norm = ModifierNode(z, dev_rng, ImageLikelihoodNormalizer | params.c_reg)
    PosteriorModel(z_norm | experiment.depth_image)
end

"""
    smooth_posterior(params, experiment, μ_node, dev_rng)
A posterior model which does calculate the pixel association probability `o`.
The pixel tail distribution is a mixture of a smoothed exponential and uniform distribution.
Provide a prior for `t, r` and the expected depth `μ` via the `μ_node`.
"""
function smooth_posterior(params, experiment, μ_node, dev_rng)
    # NOTE Analytic pixel association is only a deterministic function and not a Gibbs sampler in the traditional sense. Gibbs sampler would call rand(q(o|t,r,μ)) and not fn(μ,z). Probably "collapsed Gibbs" is the correct expression for it.
    o_fn = smooth_association_fn(params)
    # condition on data via closure
    o = DeterministicNode(:o, μ -> o_fn.(experiment.prior_o, μ, experiment.depth_image), (μ_node,))
    # NOTE ValidPixel diverges without regularization. Moreover, ValidPixel is not required if a regularization is used.
    pixel_model = smooth_mixture | (params.min_depth, params.max_depth, params.pixel_θ, params.pixel_σ)
    z = BroadcastedNode(:z, dev_rng, pixel_model, (μ_node, o))
    # NOTE seems to perform better with ImageLikelihoodNormalizer if prior is known for o. Also seems to perform worse than the simple_posterior if SimpleImageRegularization is used.
    z_norm = ModifierNode(z, dev_rng, ImageLikelihoodNormalizer | params.c_reg)
    PosteriorModel(z_norm | experiment.depth_image)
end
