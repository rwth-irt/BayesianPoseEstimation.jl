# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
Load the pose errors from disk, eagerly match the poses and calculate the average recall.
"""

# TODO make this a function similar to evaluate_errors

using DrWatson
@quickactivate("MCMCDepth")

using CSV
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

# Recall by sampler
groups = groupby(results, [:sampler])
recalls = combine(groups, :adds_thresh => (x -> recall(x...)) => :adds_recall, :vsd_thresh => (x -> recall(x...)) => :vsd_recall, :vsdbop_thresh => (x -> recall(x...)) => :vsdbop_recall)
combine(groups)
CSV.write(datadir("exp_pro", experiment_name, "sampler_recall.csv"), recalls)
display(recalls)

# Recall by sampler and dataset
groups = groupby(results, [:sampler, :dataset])
recalls = combine(groups, :adds_thresh => (x -> recall(x...)) => :adds_recall, :vsd_thresh => (x -> recall(x...)) => :vsd_recall, :vsdbop_thresh => (x -> recall(x...)) => :vsdbop_recall)
combine(groups)
CSV.write(datadir("exp_pro", experiment_name, "sampler_dataset_recall.csv"), recalls)
display(recalls)

# Plot recall over error threshold
import CairoMakie as MK
diss_defaults()

fig_recall = MK.Figure()
ax_vsd_recall = MK.Axis(fig_recall[2, 1]; xlabel="error threshold", ylabel="recall", limits=(nothing, (0, 1)), title="VSD")
ax_adds_recall = MK.Axis(fig_recall[2, 2]; xlabel="error threshold", ylabel="recall", limits=(nothing, (0, 1)), title="ADDS")
gl_recall = fig_recall[1, :] = MK.GridLayout()
ax_vsdbop_recall = MK.Axis(gl_recall[1, 1]; xlabel="error threshold", ylabel="recall", limits=(nothing, (0, 1)), title="VSDBOP")

fig_density = MK.Figure(figure_padding=10)
ax_vsd_density = MK.Axis(fig_density[2, 1]; xlabel="error value", ylabel="density", title="VSD")
ax_adds_density = MK.Axis(fig_density[2, 2]; xlabel="error value", ylabel="density", title="ADDS")
gl_density = fig_density[1, :] = MK.GridLayout()
ax_vsdbop_density = MK.Axis(gl_density[1, 1]; xlabel="error value", ylabel="density", title="VSDBOP")

θ_range = 0:0.02:1
groups = groupby(results, :sampler)
label_for_sampler = Dict("smc_mh" => "SMC-MH", "mh_sampler" => "MCMC-MH", "mtm_sampler" => "MTM")
ds = nothing
for group in groups
    adds_thresh = map(θ -> threshold_errors.(group.adds, θ), θ_range)
    adds_recalls = map(x -> recall(x...), adds_thresh)
    MK.lines!(ax_adds_recall, θ_range, adds_recalls; label=label_for_sampler[first(group.sampler)])
    MK.density!(ax_adds_density, vcat(group.adds...); label=label_for_sampler[first(group.sampler)], boundary=(0, 1))

    vsd_thresh = map(θ -> threshold_errors.(group.vsd, θ), θ_range)
    vsd_recalls = map(x -> recall(x...), vsd_thresh)
    MK.lines!(ax_vsd_recall, θ_range, vsd_recalls; label=label_for_sampler[first(group.sampler)])
    MK.density!(ax_vsd_density, vcat(group.vsd...); label=label_for_sampler[first(group.sampler)], boundary=(0, 1))

    vsdbop_thresh = map(θ -> threshold_errors.(vcat(group.vsdbop...), θ), θ_range)
    vsdbop_recalls = map(x -> recall(x...), vsdbop_thresh)
    MK.lines!(ax_vsdbop_recall, θ_range, vsdbop_recalls; label=label_for_sampler[first(group.sampler)])
    MK.density!(ax_vsdbop_density, reduce(vcat, reduce(vcat, group.vsdbop)); label=label_for_sampler[first(group.sampler)], boundary=(0, 1))
end

MK.vlines!(ax_vsdbop_recall, BOP19_THRESHOLDS)
MK.vspan!(ax_vsdbop_recall, 0, last(BOP19_THRESHOLDS))
MK.vlines!(ax_vsd_recall, BOP18_θ)
MK.vspan!(ax_vsd_recall, 0, BOP18_θ)
MK.vlines!(ax_adds_recall, ADDS_θ)
MK.vspan!(ax_adds_recall, 0, ADDS_θ)
MK.Legend(gl_recall[1, 2], ax_vsdbop_recall)
# display(fig_recall)
save(joinpath("plots", "$(experiment_name)_recall.pdf"), fig_recall)

MK.vlines!(ax_vsdbop_density, BOP19_THRESHOLDS)
MK.vspan!(ax_vsdbop_density, 0, last(BOP19_THRESHOLDS))
MK.vlines!(ax_vsd_density, BOP18_θ)
MK.vspan!(ax_vsd_density, 0, BOP18_θ)
MK.vlines!(ax_adds_density, ADDS_θ)
MK.vspan!(ax_adds_density, 0, ADDS_θ)
MK.Legend(gl_density[1, 2], ax_vsdbop_density)
# display(fig_density)
save(joinpath("plots", "$(experiment_name)_density.pdf"), fig_density)
