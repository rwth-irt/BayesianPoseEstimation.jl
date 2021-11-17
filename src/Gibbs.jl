# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

#TODO I guess implementing AbstractMCMC might be a good idea
# https://turinglang.github.io/AbstractMCMC.jl/dev/design/
# https://turing.ml/dev/docs/for-developers/interface

struct Gibbs
    model
    samplers::Vector{AbtractSampler}
end

"""
    gibbs_proposal(model, params)
"""
function gibbs_proposal(model, vars)
    # TODO randomly sample from before, predict from after
    proposal_model = Soss.likelihood(model, vars)
    predictive_model = Soss.predictive(model, vars)
    # todo merge while retaining old prior
end

function merge_gibbs_samples(s1, s2)

end