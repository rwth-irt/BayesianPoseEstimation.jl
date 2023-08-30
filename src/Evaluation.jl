# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using DataFrames

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
    combine_estimates(df_group)
Combines the df_group by sorting the errors by the ground truth for the use in `match_errors`.
Moreover, the score for the estimate is stored.

`df_group` must be grouped by the estimates, e.g via `:t`.
"""
function combine_estimates(df_group)
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
    errors_per_estimate = combine(combine_estimates, estimate_groups)
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