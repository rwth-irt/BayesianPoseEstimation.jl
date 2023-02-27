# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    render_context(params)
Generate a context from the MCMCDepth Parameters.
"""
render_context(params::Parameters) = depth_offscreen_context(params.width, params.height, params.depth, device_array_type(params))

"""
    render(render_context, scene, pose)
Renders the first object in the scene with a given pose.
Returns a matching view to the mapped render data array of the context.
"""
render(render_context::OffscreenContext, scene::Scene, pose::Pose) = draw(render_context, pose_scene(scene, pose))

"""
    render(render_context, scene, pose)
Renders the first object in the scene with a given pose.
Returns a matching view to the mapped render data array of the context.
"""
render(render_context::OffscreenContext, scene::Scene, poses::AbstractVector{<:Pose}) = draw(render_context, [pose_scene(scene, pose) for pose in poses])

# Create a scene for the given object pose
pose_scene(scene::Scene, pose::Pose) = @set first(scene.meshes).pose = pose
