# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using DataFrames
using ProgressLogging
using ThreadsX

"""
    es_pose(df_row)
Extract and convert the estimated pose from a DataFrame row.
"""
es_pose(df_row) = to_pose(df_row.t, df_row.R)

"""
    gt_pose(df_row)
Extract and convert the ground truth pose from a DataFrame row.
"""
gt_pose(df_row) = to_pose(df_row.gt_t, df_row.gt_R)

"""
    adds_row(df_row)
Calculate the ADDS error for a DataFrame row.
"""
function adds_row(df_row)
    if ismissing(df_row.t)
        # No prediction -> always wrong -> ∞ error
        Inf32
    else
        mesh_eval = load_mesh_eval(df_row)
        points = mesh_eval.position
        gt = gt_pose(df_row)
        es = es_pose(df_row)
        normalized_adds_error(points, es, gt, df_row.diameter)
    end
end

"""
    vsd_row(df_row, dist_context, δ)
Calculate the VSD (BOP18) error for a DataFrame row.
Provide a SciGL `distance_offscreen_context` and the `δ` of the dataset.
"""
function vsd_row(df_row, dist_context, δ)
    if ismissing(df_row.t)
        # No prediction -> always wrong -> ∞ error
        Inf32
    else
        mesh = load_mesh_eval(df_row)
        gt = gt_pose(df_row)
        es = es_pose(df_row)
        cv_camera = df_row.cv_camera
        width, height = size(dist_context.render_data)
        depth_img = load_depth_image(df_row, width, height)
        dist_img = depth_to_distance(depth_img, cv_camera)
        # BOP19 and later use normalized version with multiple τ
        vsd_error(dist_context, cv_camera, mesh, dist_img, es, gt; δ=δ)
    end
end

"""
    vsdbop_row(df_row, dist_context, δ)
Calculate the VSD BOP19 errors for a DataFrame row.
Provide a SciGL `distance_offscreen_context` and the `δ` of the dataset.

Returns errors for τ ∈ 0.05:0.05:0.5
"""
function vsdbop_row(df_row, dist_context, δ)
    if ismissing(df_row.t)
        # No prediction -> always wrong -> ∞ error
        return fill(Inf32, length(BOP19_THRESHOLDS))
    else
        mesh = load_mesh_eval(df_row)
        gt = gt_pose(df_row)
        es = es_pose(df_row)
        cv_camera = df_row.cv_camera
        width, height = size(dist_context.render_data)
        depth_img = load_depth_image(df_row, width, height)
        dist_img = depth_to_distance(depth_img, cv_camera)
        # BOP19 and later use normalized version with multiple τ
        normalized_vsd_error(dist_context, cv_camera, mesh, dist_img, es, gt, df_row.diameter; δ=δ, τ=BOP19_THRESHOLDS)
    end
end

"""
    combine_est_errors(df_group)
Combines the df_group by sorting the errors by the ground truth for the use in `match_errors`.
Moreover, the score for the estimate is stored.

`df_group` must be grouped by the estimates, e.g via `:t`.
"""
function combine_est_errors(df_group)
    # Order of ground truth must be the same
    sorted = sort(df_group, :gt_t)
    result = (; score=[sorted.score], gt_t=[sorted.gt_t])
    if "adds" in names(df_group)
        result = (; result..., adds=[sorted.adds])
    end
    if "vsd" in names(df_group)
        result = (; result..., vsd=[sorted.vsd])
    end
    if "vsdbop" in names(df_group)
        result = (; result..., vsdbop=[sorted.vsdbop])
    end
    result
end

"""
    match_obj_errors(df_group)
`group` must be grouped by the scene_id, image_id, and obj_id.
For this estimate, 
Greedily matches at most one ground truth to each estimated pose of the object.
If no estimate is available for a ground truth, Inf set as error.
"""
function match_obj_errors(df_group)
    estimate_groups = groupby(df_group, :t)
    errors_per_estimate = combine(combine_est_errors, estimate_groups)
    score_per_estimate = first.(errors_per_estimate.score)
    # Returns Inf for each gt where no estimate is available.
    result = (;)
    if "adds" in names(df_group)
        result = (; result..., adds=match_errors(score_per_estimate, errors_per_estimate.adds))
    end
    if "vsd" in names(df_group)
        result = (; result..., vsd=match_errors(score_per_estimate, errors_per_estimate.vsd))
    end
    if "vsdbop" in names(df_group)
        result = (; result..., vsdbop=match_bop19_errors(score_per_estimate, errors_per_estimate.vsdbop))
    end
    result
end

"""
    evaluate_errors(experiment_name)
Evaluate the errors of the pose estimates in `exp_raw/experiment_name`.
The VSD error is evaluated with a context using 100x100px crops.
"""
function evaluate_errors(experiment_name)
    dir = datadir("exp_raw", experiment_name)
    files = readdir(dir)
    configs = my_parse_savename.(files)
    parameters = load(joinpath(dir, first(files)))["parameters"]
    dist_context = distance_offscreen_context(100, 100, parameters.depth)
    calc_n_match_closure = calc_n_match_errors | (dist_context, experiment_name)
    try
        @progress "evaluating error metrics, experiment: $experiment_name" for config in configs
            @produce_or_load(calc_n_match_closure, config, datadir("exp_pro", experiment_name, "errors"); filename=my_savename)
        end
    finally
        destroy_context(dist_context)
    end
end

"""
    calc_n_match_errors(dist_context, experiment_name, config)
Calculate and match the errors for each ground truth - estimate combination (per scene_id, img_id, obj_id).
"""
function calc_n_match_errors(dist_context, experiment_name, config)
    @unpack dataset, testset, scene_id = config

    # Load the estimates for the scene
    est_directory = datadir("exp_raw", experiment_name)
    est_dict = load(joinpath(est_directory, my_savename(config, "jld2")))
    est_df = est_dict["result_df"]
    est_df = est_df[!, [:scene_id, :img_id, :obj_id, :t, :R, :score]]

    # Add gt_R & gt_t for testset
    gt_df = bop_test_or_train(dataset, testset, scene_id)
    datasubset_path = datadir("bop", dataset, testset)
    if !("gt_t" in names(gt_df))
        leftjoin!(gt_df, PoseErrors.gt_dataframe(datasubset_path, scene_id)
            ; on=[:scene_id, :img_id, :gt_id])
    end
    if !("visib_fract" in names(gt_df))
        leftjoin!(gt_df, PoseErrors.gt_info_dataframe(datasubset_path, scene_id); on=[:scene_id, :img_id, :gt_id])
    end
    df = outerjoin(gt_df, est_df; on=[:scene_id, :img_id, :obj_id])

    # Keep only visibility fraction >= 0.1
    filter!(:visib_fract => (x -> x >= 0.1), df)
    # Only estimates for which a ground truth exists are relevant for the recall
    filter!(:gt_t => (x -> !ismissing(x)), df)

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

"""
    my_parse_savename(file [;connector=","])
Broadcastable version of `DrWatson.parse_savename` which returns the config *without* prefix and suffix.
"""
function my_parse_savename(file; connector=",")
    _, config, _ = parse_savename(file; connector=connector)
    config
end

"""
    my_savename(config [, suffix="" ;connector=","])
Uses default connector ",".
"""
my_savename(config, suffix=""; connector=",") = savename(config, suffix; connector=connector)

"""
    step_time_50px(sampler, n_particles)
Calculate the time of one inference step for image crops of step_time_50px.
The model is fit in scripts/inference_time.jl
"""
function step_time_50px(sampler, n_particles)
    sampler = Symbol(sampler)
    if sampler == :mtm_sampler
        2.0713917871805535e-5 * n_particles + 0.0009689999930297939
    elseif sampler == :smc_bootstrap
        9.859881016718867e-6 * n_particles + 0.0005444291793518054
    elseif sampler == :smc_forward
        1.011467981699705e-5 * n_particles + 0.00046016282930897107
    elseif sampler == :smc_mh
        1.0213961859944646e-5 * n_particles + 0.0005127889784754975
    end
end

"""
    step_time_100px(sampler, n_particles)
Calculate the time of one inference step for image crops of 100x100px.
The model is fit in scripts/inference_time.jl
"""
function step_time_100px(sampler, n_particles)
    if n_particles > 300
        throw(DomainError(n_particles, "Model is only valid for less than 300 particles"))
    end
    sampler = Symbol(sampler)
    if sampler == :mtm_sampler
        2.3913472130279465e-5 * n_particles + 0.000811217292662409
    elseif sampler == :smc_bootstrap
        1.4303196800047856e-5 * n_particles + 0.0004177786090856081
    elseif sampler == :smc_forward
        1.4623637212412754e-5 * n_particles + 0.0003903391451899304
    elseif sampler == :smc_mh
        1.560852741922705e-5 * n_particles + 0.00046497024011949063
    end
end

"""
    step_time_200px(sampler, n_particles)
Calculate the time of one inference step for image crops of 200x200px.
The model is only valid from 1-300 particles, as fit in scripts/inference_time.jl
"""
function step_time_200px(sampler, n_particles)
    if n_particles > 300
        throw(DomainError(n_particles, "Model is only valid for less than 300 particles"))
    end
    sampler = Symbol(sampler)
    if sampler == :mtm_sampler
        3.409399935861806e-5 * n_particles + 0.0010246466794629396
    elseif sampler == :smc_bootstrap
        2.6782985105751874e-5 * n_particles + 0.0005169805241719439
    elseif sampler == :smc_forward
        2.649041367750422e-5 * n_particles + 0.0004506619640268151
    elseif sampler == :smc_mh
        3.166933852250932e-5 * n_particles + 0.0004264868482365757
    end
end