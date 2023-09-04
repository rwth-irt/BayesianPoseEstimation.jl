# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
Load the pose errors from disk, eagerly match the poses and calculate the average recall.
"""

# TODO enable plotting recall over threshold
# TODO make this a function similar to evaluate_errors

using DrWatson
@quickactivate("MCMCDepth")

using DataFrames
using MCMCDepth
using PoseErrors

# Combine results by sampler & dataset
experiment_name = "baseline"
directory = datadir("exp_pro", experiment_name, "errors")
results = collect_results(directory)
function parse_config(path)
    config = my_parse_savename(path)
    @unpack sampler, dataset, scene_id = config
    sampler, dataset, scene_id
end
transform!(results, :path => ByRow(parse_config) => [:sampler, :dataset, :scene_id])

# Threshold the errors
transform!(results, :adds => ByRow(x -> threshold_errors(x, ADDS_θ)) => :adds_thresh)
transform!(results, :vsd => ByRow(x -> threshold_errors(x, BOP18_θ)) => :vsd_thresh)
transform!(results, :vsdbop => ByRow(x -> threshold_errors(vcat(x...), BOP19_THRESHOLDS)) => :vsdbop_thresh)

# Recall by sampler and/or dataset
groups = groupby(results, [:sampler])
# groups = groupby(results, [:sampler, :dataset])
recalls = combine(groups, :adds_thresh => (x -> recall(x...)) => :adds_recall, :vsd_thresh => (x -> recall(x...)) => :vsd_recall, :vsdbop_thresh => (x -> recall(x...)) => :vsdbop_recall)
combine(groups,)

display(recalls)
