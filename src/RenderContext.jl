# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using PoseErrors

"""
    render_fn(render_context, scene, t, r)
Function can be conditioned on the `render_context` and `scene` to be used in a model node to render different poses for t & r.
"""
function render_fn(render_context, scene, t, r)
    p = to_pose(t, r)
    render(render_context, scene, p)
end

"""
    render_crop_fn(render_context, scene, object_id, t, r)
Function can be conditioned on the `render_context`, `scene`, and object `diameter` to be used in a model node to render different poses for t & r.
The images will be cropped to the center of the object and 1.5x the diameter.
"""
function render_crop_fn(render_context, scene::Scene, diameter, t, r)
    crop_cam = crop(scene.camera.object, t, diameter)
    p = to_pose(t, r)
    render(render_context, Scene(crop_cam, scene.meshes), p)
end

# Assumes that all positions are close
SciGL.crop(camera::CvCamera, centers::AbstractMatrix, diameter) = crop(camera, centers[:, 1], diameter)

# BUG depth_offscreen_context with CUDA OpenGL interop does not work on multi GPU system
"""
    render_context(params)
Generate a context from the MCMCDepth Parameters.
"""
render_context(params::Parameters) = depth_copy_offscreen_context(params.width, params.height, params.depth, device_array_type(params))

"""
    render(render_context, scene, pose)
Renders the first object in the scene with a given pose.
Returns a matching view to the mapped render data array of the context.
"""
render(render_context, scene::Scene, pose::Pose) = draw(render_context, pose_scene(scene, pose))

"""
    render(render_context, scene, pose)
Renders the first object in the scene with a given pose.
Returns a matching view to the mapped render data array of the context.
"""
render(render_context, scene::Scene, poses::AbstractVector{<:Pose}) = draw(render_context, [pose_scene(scene, pose) for pose in poses])

# Create a scene for the given object pose
pose_scene(scene::Scene, pose::Pose) = @set first(scene.meshes).pose = pose
