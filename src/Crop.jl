# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# TODO remove
using CoordinateTransformations
using ImageTransformations: imresize
using Interpolations: Constant
using Rotations

# Compared to SciGL no conversion between OpenGL and OpenCV conventions required

cv_intrinsic(camera::CvCamera) = LinearMap([
    camera.f_x camera.s camera.c_x
    0 camera.f_y camera.c_y
    0 0 1
])

cv_view(camera_pose) = inv(AffineMap(camera_pose))

cv_transform(camera::CvCamera, camera_pose::Pose=one(Pose)) = cv_intrinsic(camera) ∘ cv_view(camera_pose)

function cv_project(camera::CvCamera, point3d::AbstractVector, camera_pose::Pose=one(Pose))
    uv = cv_transform(camera, camera_pose)(point3d)
    uv[1:2] ./ uv[3]
end

# TODO mind julia convention, shift it one pixel right and down
function crop_center(camera::CvCamera, center3d::AbstractVector, camera_pose::Pose=one(Pose))
    transformed = cv_project(camera, center3d, camera_pose)
    center2d = round.(Int, transformed[1:2])
    # OpenCV assumes (0,0) as the origin, while Julia arrays start at (1,1)
    center2d .+ 1
end

"""
    clamp_boundingbox(left, right, top, bottom, width, height)
Clamp the corners of the bounding box (left, right, top, bottom) to the camera's image dimensions (0, 0, width, height).
"""
function clamp_boundingbox(left, right, top, bottom, width, height)
    left = max(left, zero(left))
    right = min(right, width)
    top = max(top, zero(top))
    bottom = min(bottom, height)
    left, right, top, bottom
end

"""
    crop_boundingbox(camera, center3d, model_diameter, [camera_pose=one(Pose)])
Returns the parameters (left, right, top, bottom) of the bounding box to crop the image to.
"""
function crop_boundingbox(camera::CvCamera, center3d::AbstractVector, model_diameter, camera_pose::Pose=one(Pose))
    center2d = crop_center(camera, center3d, camera_pose)
    diameter3d = 1.5 * model_diameter
    width, height = @. ceil(Int, (camera.f_x, camera.f_y) * diameter3d / center3d[3])
    left, top = @. floor(Int, center2d - (width, height) / 2)
    right, bottom = left + width, top + height
    clamp_boundingbox(left, right, top, bottom, camera.width, camera.height)
end

"""
    crop_img(img, left, right, top, bottom)    
    crop_img(img, (left, right, top, bottom))
Convenience method to create a view of the image for the bounding box coordinates.    
"""
crop_image(img, left, right, top, bottom) = @view img[left:right, top:bottom]

"""
    depth_resize(img, args...; kwargs...)
Avoids interpolations when resizing depth images.
What might look nice on color images causes wrong values on depth images, since no interpolated values in between two discontinuous surfaces would be captured by a real camera.
Calls ImageTransformations.jl `imresize(img, args...; kwargs..., method=Constant())`
"""
depth_resize(img, args...; kwargs...) = imresize(img, args...; kwargs..., method=Constant())
