# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Base.Filesystem
using FileIO
using ImageIO
using Images
using JSON

dataset_path(name) = joinpath(pwd(), "datasets", name)
test_path(name) = joinpath(dataset_path(name), "test")
test_scenes(name) = readdir(test_path(name); join=true)
lpad_bop(number) = lpad(number, 6, "0")
first_scene_path = test_scenes("lm") |> first

# Per scene
scene_cam = JSON.parsefile(joinpath(first_scene_path, "scene_camera.json"))
scene_gt = JSON.parsefile(joinpath(first_scene_path, "scene_gt.json"))

# Per image
img_number, gt_data = first(scene_gt)
img_cam = scene_cam[img_number]
depth_scale = img_cam["depth_scale"]
cam_K = reshape(img_cam["cam_K"], 3, 3)

# load the image
depth_dir = joinpath(first_scene_path, "depth")
img_names = readdir(depth_dir; join=true)
img_numbers = @. parse(Int, img_names |> splitext |> first |> splitpath |> last)
img_dict = Dict(img_numbers .=> img_names)
img_path = img_dict[parse(Int, img_number)]
depth_img = load(img_path);
# TODO cleaner?
depth_img_m = 1e-3 .* reinterpret.(UInt16, channelview(depth_img))
# TODO looks plausible?
Gray.(depth_img_m)
Gray.(depth_img_m ./ maximum(depth_img_m))

# Per object
obj_id = first(gt_data)["obj_id"]
rotation = reshape(first(gt_data)["cam_R_m2c"], 3, 3)
translation = 1e-3 * first(gt_data)["cam_t_m2c"]

# TODO load the object

# TODO Goal: load an element from scene_gt.json and render the gt pose on top of the image.