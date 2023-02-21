# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Base.Filesystem
using FileIO
using ImageCore
using ImageIO
using JSON
using SciGL

dataset_path(dataset_name) = joinpath(pwd(), "datasets", dataset_name)
datasubset_path(dataset_name, subset_name="test") = joinpath(dataset_path(dataset_name), subset_name)

"""
    scene_paths(dataset_name, [subset_name="test"])
Returns a vector of the full paths to the scene directories of the datasets subset.
"""
scene_paths(dataset_name="lm", subset_name="test") = readdir(datasubset_path(dataset_name, subset_name); join=true)

"""
    lpad_bop(number)
Pads the number with zeros from the left for a total length of six digits.
`pad_bop(42) = 000042`
"""
lpad_bop(number) = lpad(number, 6, "0")

"""
    image_cameras(scene_path, image_sizes)
Load the scene's cameras as a Dictionary of the SciGL.CvCamera with the image numbers as keys.
`image_sizes` is a dictionary with the image numbers as keys and the size as Tuple `(width, height)`.
"""
function image_cameras(scene_path, image_sizes)
    json_cams = JSON.parsefile(joinpath(scene_path, "scene_camera.json"))
    result = Dict{Int,CvCamera}()
    for (key, value) in json_cams
        key = parse(Int, key)
        width, height = image_sizes[key]
        cam_K = value["cam_K"] .|> Float32
        cv_cam = CvCamera(width, height, cam_K[1], cam_K[5], cam_K[3], cam_K[6]; s=cam_K[4])
        result[key] = cv_cam
    end
    result
end

"""
    depth_image_scales(scene_path)
Load the scene's depth scales as a Dictionary of Float64 with the image numbers as keys.
"""
function depth_image_scales(scene_path)
    json_cams = JSON.parsefile(joinpath(scene_path, "scene_camera.json"))
    result = Dict{Int,Float64}()
    for (key, value) in json_cams
        depth_scale = value["depth_scale"] .|> Float64
        result[parse(Int, key)] = depth_scale
    end
    result
end

"""
    image_paths(scene_path, modality="depth")
Load the image paths as Dictionary of Strings with the image numbers as keys.
"""
function image_paths(scene_path, modality="depth")
    dir = joinpath(scene_path, modality)
    img_names = readdir(dir; join=true)
    img_numbers = @. parse(Int, img_names |> splitext |> first |> splitpath |> last)
    Dict(img_numbers .=> img_names)
end

"""
    image_paths(scene_path, modality="depth")
Load the image sizes as Dictionary of Dims{2} with the image numbers as keys.
"""
function image_sizes(image_paths)
    img_sizes = Dict{Int,Dims{2}}()
    for (key, value) in img_paths
        img = load(value)
        # Julia images are transposed
        height, width = size(img)
        img_sizes[key] = (width, height)
    end
    img_sizes
end

# ImageIO loads image transposed by default
load_depth_image(path, scale) = Float32(1e-3 * scale) .* rawview(channelview(load(path)))'
"""
    load_depth_image(number, paths, scales)
Load the depth image as a Matrix{Float32} of size (width, height) where each pixel is the depth in meters.
"""
load_depth_image(number, paths, scales) = load_depth_image(paths[number], scales[number])

# Per scene
first_scene_path = scene_paths("lm") |> first

# Per image
img_paths = image_paths(first_scene_path, "depth")
img_sizes = image_sizes(first_scene_path)
img_cams = image_cameras(first_scene_path, img_sizes)
img_scales = depth_image_scales(first_scene_path)

# load the image
img_number = parse(Int, first(scene_gt |> keys))
depth_img = load_depth_image(img_number, img_paths, img_scales)

Gray.(depth_img)
Gray.(depth_img ./ maximum(depth_img))

# Per evaluation
scene_gt = JSON.parsefile(joinpath(first_scene_path, "scene_gt.json"))
obj_id = first(gt_data)["obj_id"]
rotation = reshape(first(gt_data)["cam_R_m2c"], 3, 3)
translation = 1e-3 * first(gt_data)["cam_t_m2c"]

# TODO load as data frame

# TODO Goal: load an element from scene_gt.json and render the gt pose on top of the image.