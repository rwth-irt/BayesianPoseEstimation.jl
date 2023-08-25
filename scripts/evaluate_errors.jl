# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
Load the results from disk and evaluate pose error metrics.
"""

using DrWatson
@quickactivate("MCMCDepth")

using DataFrames
using MCMCDepth
using PoseErrors
using SciGL
using ThreadsX

using ProgressLogging
using TerminalLoggers
using Logging: global_logger
global_logger(TerminalLogger(right_justify=120))

"""
    load_df(directory, file)
Loads the experiments result as a DataFrame.
"""
function load_df(directory, file)
    dict = load(joinpath(directory, file))
    experiment_df = dict["result_df"]
    # Keep only relevant columns
    experiment_df = experiment_df[!, [:scene_id, :img_id, :obj_id, :t, :R, :score, :time]]

    # Load corresponding gt data
    _, config = parse_savename(file; connector=",")
    @unpack dataset, testset, scene_id = config
    gt_df = bop_test_or_train(dataset, testset, scene_id)

    # Add gt_R & gt_t for testset
    datasubset_path = datadir("bop", dataset, testset)
    if !("gt_t" in names(gt_df))
        leftjoin!(gt_df, PoseErrors.gt_dataframe(datasubset_path, scene_id)
            ; on=[:scene_id, :img_id, :gt_id])
    end
    if !("visib_fract" in names(gt_df))
        leftjoin!(gt_df, PoseErrors.gt_info_dataframe(datasubset_path, scene_id); on=[:scene_id, :img_id, :gt_id])
    end

    # Match estimates to ground truths
    joined = outerjoin(gt_df, experiment_df; on=[:scene_id, :img_id, :obj_id])
    # Ignore visib_fract < 0.1
    filter!(row -> row.visib_fract >= 0.1, joined)
    # Estimates without ground truth are not relevant for recall
    filter!(:gt_t => (x -> !ismissing(x)), joined)
end

"""
    calc_n_match_errors(dist_context, scene_df, config)
Calculate and match the errors for each ground truth - estimate combination (per scene_id, img_id, obj_id).
"""
function calc_n_match_errors(dist_context, experiment_name, config)
    @unpack dataset = config
    # Load the estimates for the scene
    directory = datadir("exp_raw", experiment_name)
    file = savename(config, "jld2"; connector=",")
    df = load_df(directory, file)

    # Calculate different error metrics
    # Different VSD δ for visible surface in ITODD & steri
    vsd_δ = contains(dataset, "itodd") || contains(dataset, "steri") ? ITODD_δ : BOP_δ |> Float32
    # WARN do not parallelize using ThreadsX, OpenGL is sequential
    df.vsd = map(row -> vsd_row(row, dist_context, vsd_δ), eachrow(df))
    df.vsdbop = map(row -> vsdbop_row(row, dist_context, vsd_δ), eachrow(df))
    # This is ~10x faster using ThreadsX
    df.adds = ThreadsX.map(adds_row, eachrow(df))

    # Greedy matching of the ground truth to the estimates
    errors_per_obj = groupby(df, [:scene_id, :img_id, :obj_id])
    matched_df = combine(match_obj_errors, errors_per_obj; threads=false)
    # must be the number of ground truth annotations
    @assert nrow(matched_df) == nrow(unique(df, :gt_t))

    Dict("vsd" => matched_df.vsd, "vsdbop" => matched_df.vsdbop, "adds" => matched_df.adds)
end

# TODO iterate over directorie in exp_ra
exp_raw = datadir("exp_raw")
experiments = readdir(exp_raw)
@progress "evaluating errors" for experiment_name in experiments
    # Collect the results per experiment / scene
    directory = datadir("exp_raw", experiment_name)
    files = readdir(directory)
    parameters = load(joinpath(directory, first(files)))["parameters"]
    # TODO The size of the context does matter because of scaling & interpolation of the crop
    dist_context = distance_offscreen_context(parameters.width, parameters.height, parameters.depth)

    @progress "evaluating $experiment_name" for file in files
        _, config = parse_savename(file; connector=",")
        calc_match_closure(config) = calc_n_match_errors(dist_context, experiment_name, config)
        @produce_or_load(calc_match_closure, config, datadir("exp_pro", experiment_name, "errors"); filename=c -> savename(c; connector=","))
    end

    destroy_context(dist_context)
end
