# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using CUDA
using ModernGL
using SciGL

"""
    RenderContext
Stores all the static information required for rendering the object in the scene (everything except the latent variables like pose and occlusion).
Use it for offscreen rendering to the texture attachment of the `framebuffer`.
The `render_data` maps the storage of the `gl_buffer` to an (Cu)Array which keeps the memory consumption constant.
Thus, copy the data from the `framebuffer` to the `gl_buffer` to update the `render_data`.
Either use the synchronous `unsafe_copyto!` or use `async_copyto!` and `sync_buffer` if you can execute calculation in the meantime.
"""
struct RenderContext{T,F<:GLAbstraction.FrameBuffer,C<:AbstractArray{T},P<:GLAbstraction.AbstractProgram}
    window::GLFW.Window
    # Preallocate a CPU Array or GPU CuArray, this also avoids having to pass a device flag
    framebuffer::F
    gl_buffer::PersistentBuffer{T}
    render_data::C
    shader_program::P
end

"""
    RenderContext(width, height, depth, [T=Array])
Simplified generation of an OpenGL context for rendering depth images of a specific size.
Batched rendering is enabled by generating a 3D Texture of Float32 with size (width, height, layer).
Specify the `render_data` type as `Array` vs. `CuArray` to choose your compute device for the inference calculations.
"""
function RenderContext(width::Integer, height::Integer, depth::Integer, ::Type{T}=Array) where {T<:AbstractArray}
    window = context_offscreen(width, height)
    # RBO supports only 2D, Texture 3D for rendering multiple samples
    framebuffer = depth_framebuffer(width, height, depth)
    texture = first(GLAbstraction.color_attachments(framebuffer))
    # Store depth values as Float32 to avoid conversion from Gray
    gl_buffer = PersistentBuffer(Float32, texture)
    render_data = T(gl_buffer)
    program = GLAbstraction.Program(SimpleVert, DepthFrag)
    enable_depth_stencil()
    set_clear_color()
    RenderContext(window, framebuffer, gl_buffer, render_data, program)
end

"""
    RenderContext(params, [T=Array])
Generate a context from the MCMCDepth Parameters.
"""
RenderContext(params::Parameters, T::Type{<:AbstractArray}) = RenderContext(params.width, params.height, params.depth, T)

Base.show(io::IO, context::RenderContext{T}) where {T} = print(io, "RenderContext{$(T)}\n$(context.framebuffer)\nRender Data: $(typeof(context.render_data))")

Base.size(render_context::RenderContext) = size(render_context.render_data)

"""
    render(render_context, scene, object_id, pose, layer_id)
Render the scene with the given pose for the object to the layer of the framebuffer of the context.
"""
function render(render_context::RenderContext, scene::Scene, object_id::Integer, pose::Pose, layer_id::Integer)
    # Draw to framebuffer
    GLAbstraction.bind(render_context.framebuffer)
    scene_pose = @set scene.meshes[object_id].pose = pose
    activate_layer(render_context.framebuffer, layer_id)
    clear_buffers()
    draw(render_context.shader_program, scene_pose)
    GLAbstraction.unbind(render_context.framebuffer)
end

"""
    render(render_context, scene, object_id, pose)
Renders the object with a given pose in the scene.
Returns a matching view to the mapped render data array of the context.
"""
function render(render_context::RenderContext, scene::Scene, object_id::Integer, pose::Pose)
    # Render single pose to the first layer
    render(render_context, scene, object_id, pose, 1)
    width, height = size(render_context.gl_buffer)
    unsafe_copyto!(render_context.gl_buffer, render_context.framebuffer, width, height)
    @view render_context.render_data[:, :, 1]
end

"""
    render(render_context, scene, object_id, pose)
Renders the object with a given set of poses in the scene.
Returns a matching view to the mapped render data array of the context.
"""
function render(render_context::RenderContext, scene::Scene, object_id::Integer, poses::AbstractVector{<:Pose})
    # Apply each pose to the immutable scene and render the pose to layer number idx 
    for (idx, pose) in enumerate(poses)
        render(render_context, scene, object_id, pose, idx)
    end
    width, height = size(render_context.gl_buffer)
    depth = length(poses)
    # WARN According to Stackoverflow CUDA and OpenGL do run concurrently
    # Copy only the rendered poses for performance and return a matching view, so broadcasting ignores the other (old) render_data
    unsafe_copyto!(render_context.gl_buffer, render_context.framebuffer, width, height, depth)
    @view render_context.render_data[:, :, 1:depth]
end

"""
    Scene(params, render_context)
Generate a Scene for a given set of `Parameters` and `RenderContext` .
"""
function SciGL.Scene(params::Parameters, render_context::RenderContext)
    camera = CvCamera(params.width, params.height, params.f_x, params.f_y, params.c_x, params.c_y; near=params.min_depth, far=params.max_depth) |> SceneObject
    meshes = map(params.mesh_files) do file
        load_mesh(render_context.shader_program, file) |> SceneObject
    end
    Scene(camera, meshes)
end
