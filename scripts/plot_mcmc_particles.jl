# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using DrWatson
@quickactivate "MCMCDepth"
using DataFrames
using MCMCDepth
using Plots
using PoseErrors
using Statistics
gr()
diss_defaults()

# MTM
function parse_config(path)
    config = my_parse_savename(path)
    @unpack pose_time, n_particles, sampler = config
    pose_time, n_particles, sampler
end
# Calculate recalls
pro_df = collect_results(datadir("exp_pro", "recall_n_particles", "errors"); rinclude=[r"mtm"])
transform!(pro_df, :path => ByRow(parse_config) => [:pose_time, :n_particles, :sampler])
filter!(x -> x.n_particles > 1, pro_df)
# Threshold errors
transform!(pro_df, :adds => ByRow(x -> threshold_errors(x, ADDS_θ)) => :adds_thresh)
transform!(pro_df, :vsd => ByRow(x -> threshold_errors(x, BOP18_θ)) => :vsd_thresh)
transform!(pro_df, :vsdbop => ByRow(x -> threshold_errors(vcat(x...), BOP19_THRESHOLDS)) => :vsdbop_thresh)
groups = groupby(pro_df, [:sampler, :pose_time, :n_particles])
recalls = combine(groups, :adds_thresh => (x -> recall(x...)) => :adds_recall, :vsd_thresh => (x -> recall(x...)) => :vsd_recall, :vsdbop_thresh => (x -> recall(x...)) => :vsdbop_recall)
# Calculate mean pose inference times
raw_df = collect_results(datadir("exp_raw", "recall_n_particles"); rinclude=[r"mtm"])
transform!(raw_df, :path => ByRow(parse_config) => [:pose_time, :n_particles, :sampler])
filter!(x -> x.n_particles > 1, raw_df)
groups = groupby(raw_df, [:sampler, :pose_time, :n_particles])
times = combine(groups, :time => (x -> mean(vcat(x...))) => :mean_time)
sort!(recalls, [:n_particles, :pose_time])
sort!(times, [:n_particles, :pose_time])

# MH
mh_df = collect_results(datadir("exp_pro", "recall_n_steps", "errors"))
function parse_mh_config(path)
    config = my_parse_savename(path)
    @unpack n_steps, dataset = config
    n_steps, dataset
end
transform!(mh_df, :path => ByRow(parse_mh_config) => [:n_steps, :dataset])
# Threshold errors
transform!(mh_df, :adds => ByRow(x -> threshold_errors(x, ADDS_θ)) => :adds_thresh)
transform!(mh_df, :vsd => ByRow(x -> threshold_errors(x, BOP18_θ)) => :vsd_thresh)
transform!(mh_df, :vsdbop => ByRow(x -> threshold_errors(vcat(x...), BOP19_THRESHOLDS)) => :vsdbop_thresh)
# Recall & time by n_steps
mh_groups = groupby(mh_df, :n_steps)
mh_recalls = combine(mh_groups, :adds_thresh => (x -> recall(x...)) => :adds_recall, :vsd_thresh => (x -> recall(x...)) => :vsd_recall, :vsdbop_thresh => (x -> recall(x...)) => :vsdbop_recall)
# Mean inference time
mh_raw_df = collect_results(datadir("exp_raw", "recall_n_steps"))
transform!(mh_raw_df, :path => ByRow(parse_mh_config) => [:n_steps, :dataset])
mh_groups = groupby(mh_raw_df, [:n_steps])
mh_times = combine(mh_groups, :time => (x -> mean(vcat(x...))) => :mean_time)
sort!(mh_recalls, :n_steps)
sort!(mh_times, :n_steps)

# Visualize per n_particles
recall_groups = groupby(recalls, :n_particles)
time_groups = groupby(times, :n_particles)
# Lines   
p_adds = plot(; xlabel="pose inference time / s", ylabel="ADDS recall", ylims=[0, 1], linewidth=1.5)
for (rec, tim) in zip(recall_groups, time_groups)
    plot!(p_adds, tim.mean_time, rec.adds_recall; legend=false)
end
plot!(p_adds, mh_times.mean_time, mh_recalls.adds_recall; legend=false)
vline!([0.5]; label=nothing, color=:black, linestyle=:dash, linewidth=1.5)

p_vsd = plot(; xlabel="pose inference time / s", ylabel="VSD recall", ylims=[0, 1], linewidth=1.5)
for (rec, tim) in zip(recall_groups, time_groups)
    plot!(p_vsd, tim.mean_time, rec.vsd_recall; legend=false)
end
plot!(p_vsd, mh_times.mean_time, mh_recalls.vsd_recall; legend=false)
vline!([0.5]; label=nothing, color=:black, linestyle=:dash, linewidth=1.5)

p_vsdbop = plot(; xlabel="pose inference time / s", ylabel="VSDBOP recall", ylims=[0, 1], linewidth=1.5)
for (rec, tim) in zip(recall_groups, time_groups)
    plot!(p_vsdbop, tim.mean_time, rec.vsdbop_recall; legend=:outerright, label="$(rec.n_particles |> first) particles")
end
plot!(p_vsdbop, mh_times.mean_time, mh_recalls.vsdbop_recall; label="MCMC-MH")
vline!([0.5]; label=nothing, color=:black, linestyle=:dash, linewidth=1.5)

lay = @layout [a; b c]
p = plot(p_vsdbop, p_adds, p_vsd; layout=lay)
display(p)

savefig(p, joinpath("plots", "recall_steps_mtm_mcmc.pdf"))
