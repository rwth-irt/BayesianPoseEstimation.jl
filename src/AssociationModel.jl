# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    pixel_association(dist_is, dist_not, prior, μ, z)
Consists of a distribution `dist_is(μ)` for the probability of a pixel belonging to the object of interest and `dist_not(μ)` which models the probability of the pixel not belonging to this object.
Moreover, a `prior` is required for the association probability `o`.
The `logdensityof` the observation `z` is calculated analytically by marginalizing the two distributions.
"""
function pixel_association(dist_is, dist_not, prior, μ, z)
    # Internal ValidPixels handle outliers by returning 1.0 as probability which will result in the prior q without too much overhead
    p_is = pdf(dist_is(μ), z)
    p_not = pdf(dist_not(μ), z)
    nominator = prior * p_is
    # Marginalize Bernoulli distributed by summing out o
    marginal = nominator + (1 - prior) * p_not
    # Normalized posterior
    nominator / marginal
end
