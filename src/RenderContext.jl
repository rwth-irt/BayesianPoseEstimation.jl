# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    render_context(params)
Generate a context from the MCMCDepth Parameters.
"""
render_context(params::Parameters) = depth_offscreen_context(params.width, params.height, params.depth, device_array_type(params))

"""
    render(render_context, scene, object_id, pose)
Renders the object with a given pose in the scene.
Returns a matching view to the mapped render data array of the context.
"""
render(render_context::OffscreenContext, scene::Scene, object_id::Integer, pose::Pose) = draw(render_context, pose_scene(scene, object_id, pose))

"""
    render(render_context, scene, object_id, pose)
Renders the object with a given set of poses in the scene.
Returns a matching view to the mapped render data array of the context.
"""
render(render_context::OffscreenContext, scene::Scene, object_id::Integer, poses::AbstractVector{<:Pose}) = draw(render_context, [pose_scene(scene, object_id, pose) for pose in poses])

"""
    Scene(params, render_context)
Generate a Scene for a given set of `Parameters` and an `OffscreenContext` .
"""
function SciGL.Scene(params::Parameters, render_context::OffscreenContext)
    camera = CvCamera(params.width, params.height, params.f_x, params.f_y, params.c_x, params.c_y; near=params.min_depth, far=params.max_depth)
    meshes = map(params.mesh_files) do file
        load_mesh(render_context.shader_program, file)
    end
    Scene(camera, meshes)
end

# Create a scene for the given object pose
pose_scene(scene::Scene, object_id::Integer, pose::Pose) = @set scene.meshes[object_id].pose = pose
