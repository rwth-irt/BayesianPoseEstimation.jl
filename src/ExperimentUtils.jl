# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using DrWatson

"""
    bop_test_or_train(dataset, testset, scene_id)
Either loads `test_targets` or `train_targets` depending on the testset name.
`train_targets` does not use the *test_targets_bop19.json*.
"""
function bop_test_or_train(dataset, testset, scene_id)
    bop_full_path = datadir("bop", dataset, testset)
    if occursin("test", testset)
        scene_df = test_targets(bop_full_path, scene_id)
    elseif occursin("train", testset) || occursin("val", testset)
        scene_df = train_targets(bop_full_path, scene_id)
    end
end

"""
    load_img_mesh(df_row, parameters, gl_context)
Load the depth and mask images as well as the mesh of the object for an experiment row (scene_id, image_id, obj_id). 
"""
function load_img_mesh(df_row, parameters, gl_context)
    depth_img = load_depth_image(df_row, parameters.img_size...) |> device_array_type(parameters)
    mask_img = load_mask_image(df_row, parameters.img_size...) |> device_array_type(parameters)
    mesh = upload_mesh(gl_context, load_mesh(df_row))
    depth_img, mask_img, mesh
end

"""
    collect_variables(samples, var_names)
Select a subset of the variables from the chain to save memory / storage.
"""
collect_variables(samples::AbstractVector{<:Sample}, var_names) = [@set s.variables = s.variables[var_names] for s in samples]

collect_variables(state::SmcState, var_names) = @set state.sample.variables = state.sample.variables[var_names]
