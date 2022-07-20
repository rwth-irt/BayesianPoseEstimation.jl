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
The `render_cache` maps the storage of the `gl_buffer` to an (Cu)Array.
Thus, copy the data from the `framebuffer` to the `gl_buffer` to update the `render_cache`.
Either use the synchronous `unsafe_copyto!` or use `async_copyto!` and `sync_buffer` if you can execute calculation in the meantime.
"""
struct RenderContext{T,F<:GLAbstraction.FrameBuffer,C<:AbstractArray{T},P<:GLAbstraction.AbstractProgram}
    window::GLFW.Window
    # Preallocate a CPU Array or GPU CuArray, this also avoids having to pass a device flag
    framebuffer::F
    # TODO benchmark whether inference or copy time dominates -> is double buffering worth it?
    gl_buffer::PersistentBuffer{T}
    render_data::C
    shader_program::P
end

"""
    RenderContext(width, height, batch_size, [T=Array, prog=DepthProgram])
Simplified generation of an OpenGL context for rendering depth images of a specific size.
Batched rendering is enabled by generating a 3D Texture of Float32 with size (width, height, layer).
Specify the `render_cache` type as `Array` vs. `CuArray` to choose your compute device for the inference calculations.
"""
function RenderContext(width::Integer, height::Integer, batch_size::Integer, ::Type{T}=Array, prog=GLAbstraction.Program(SimpleVert, DepthFrag)) where {T<:AbstractArray}
    window = context_offscreen(width, height)
    # RBO supports only 2D, Texture 3D for rendering multiple samples
    framebuffer = depth_framebuffer(width, height, batch_size)
    texture = first(framebuffer.attachments)
    # Store depth values as Float32 to avoid conversion from Gray
    gl_buffer = PersistentBuffer(Float32, texture)
    render_data = T(gl_buffer)
    enable_depth_stencil()
    set_clear_color()
    RenderContext(window, framebuffer, gl_buffer, render_data, prog)
end

"""
    render(context, positions, orientations)
Renders the positions and orientations and returns a matching view to the mapped `render_data`.
"""
function render(context::RenderContext, scene::Scene, object_id::Integer, poses::AbstractVector{<:Pose})
    # Apply each pose to the immutable scene and render the pose to layer number idx 
    for (idx, pose) in poses
        scene_idx = @set scene.meshes[object_id].pose = pose
        activate_layer(context.framebuffer, idx)
        clear_buffers()
        draw(context.program, scene_idx)
    end
    width, height = size(context.gl_buffer)
    depth = length(poses)
    # TODO or can I execute any calculations while using async_copyto!?
    # Copy only the rendered poses for performance and return a matching view, so broadcasting ignores the other (old) render_data
    unsafe_copyto!(context.gl_buffer, context.framebuffer, width, height, depth)
    @view context.render_data[:, :, 1:depth]
end

