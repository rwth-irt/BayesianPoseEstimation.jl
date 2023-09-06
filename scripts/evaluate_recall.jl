# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
Load the pose errors from disk, eagerly match the poses and calculate the average recall.
"""

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

# Plot recall over error threshold
import CairoMakie as MK
diss_defaults()

fig = MK.Figure(figure_padding=10)
ax_vsd = MK.Axis(fig[2, 1]; xlabel="error threshold", ylabel="VSD recall")
ax_adds = MK.Axis(fig[2, 2]; xlabel="error threshold", ylabel="ADDS recall")
ga = fig[1, :] = MK.GridLayout()
ax_vsdbop = MK.Axis(ga[1, 1]; xlabel="error threshold", ylabel="VSDBOP recall")

θ_range = 0:0.02:1
groups = groupby(results, :sampler)
label_for_sampler = Dict("smc_mh" => "SMC-MH", "mh_sampler" => "MCMC-MH", "mtm_sampler" => "MTM")
for group in groups
    adds_thresh = map(θ -> threshold_errors.(group.adds, θ), θ_range)
    adds_recalls = map(x -> recall(x...), adds_thresh)
    # MK.lines!(ax_adds, θ_range, adds_recalls; label=label_for_sampler[first(group.sampler)])
    MK.density!(ax_adds, vcat(group.adds...))

    vsd_thresh = map(θ -> threshold_errors.(group.vsd, θ), θ_range)
    vsd_recalls = map(x -> recall(x...), vsd_thresh)
    # MK.lines!(ax_vsd, θ_range, vsd_recalls; label=label_for_sampler[first(group.sampler)])
    MK.density!(ax_vsd, vcat(group.vsd...))

    vsdbop_thresh = map(θ -> threshold_errors.(vcat(group.vsdbop...), θ), θ_range)
    vsdbop_recalls = map(x -> recall(x...), vsdbop_thresh)
    # MK.lines!(ax_vsdbop, θ_range, vsdbop_recalls; label=label_for_sampler[first(group.sampler)])
    MK.density!(ax_adds, reduce(vcat, group.vsdbop))
end
MK.vlines!(ax_vsd, BOP18_θ; color=:black, linestyle=:dash)
MK.vlines!(ax_adds, ADDS_θ; color=:black, linestyle=:dash)
MK.vlines!(ax_vsdbop, [first(BOP19_THRESHOLDS), last(BOP19_THRESHOLDS)]; color=:black, linestyle=:dash)

MK.Legend(ga[1, 2], ax_vsdbop)
display(fig)
save(joinpath("plots", "$(experiment_name)_recall.pdf"), fig)
