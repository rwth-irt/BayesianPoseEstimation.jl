# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 
using GLAbstraction, GLFW, SciGL
# TODO How to avoid adding them (automatically install / use with SciGL.jl)
using CoordinateTransformations, Rotations

"""
    init_render_context(width, height)
Initializes the OpenGL context for rendering depth images off screen with the given size.

Returns a tuple `(window, framebuffer, depth_prog)` containing the GLFW Window context, the framebuffer to render depth images to and the shader program.
"""
function init_render_context(width = 150, height = 150, n_chains = 1)
    window = context_offscreen(width, height)
    tiles = Tiles(width, height, n_chains)
    framebuffer = depth_framebuffer(size(tiles)...)
    enable_depth_stencil()
    set_clear_color()
    depth_prog = GLAbstraction.Program(SimpleVert, DepthFrag)
    window, framebuffer, depth_prog
end

"""
    render_depth(framebuffer, shader, scene)
Renders a depth image and loads it into the CPU memory.
Returns the depth image in OpenGL convention, which requires `transpose(iamge[:, end:-1:1])` for correct display
"""
function render_to_cpu(framebuffer, shader, scene)
    GLAbstraction.bind(framebuffer)
    clear_buffers()
    draw(shader, scene)
    # load float values
    channelview(gpu_data(framebuffer, 1))
end

"""
    render_pose(framebuffer, shader, scene, object, t, r)
Renders the object given a pose as a translation `t` and a rotation `r` in the given scene.
"""
function render_pose(framebuffer, shader, scene, object, t, r)
    if !(object in scene.meshes)
        scene = deepcopy(scene)
        push!(scene.meshes, object)
    end
    object.pose.t = Translation(t)
    object.pose.R = RotXYZ(r...)
    render_to_cpu(framebuffer, shader, scene)
end

"""
    render_pose!(framebuffer, shader, scene, object, t, r)
Renders the object given a pose as a translation `t` and a rotation `r` in the given scene.
If object is not part of the scene yet, the scene gets modified
"""
function render_pose!(framebuffer, shader, scene, object, t, r)
    if !(object in scene.meshes)
        push!(scene.meshes, object)
    end
    object.pose.t = Translation(t)
    object.pose.R = RotXYZ(r...)
    render_to_cpu(framebuffer, shader, scene)
end

"""
    destroy_render_context
Frees the OpenGL resources.
"""
function destroy_render_context(window)
    GLFW.DestroyWindow(window)
end

"""
    main()
The main inference script
"""
function main()
    # Acquire OpenGL resources
    window, framebuffer, depth_prog = init_render_context()

    # Load the scene
    camera = CvCamera(WIDTH, HEIGHT, 1.2 * WIDTH, 1.2 * HEIGHT, WIDTH / 2, HEIGHT / 2) |> SceneObject
    monkey = load_mesh(depth_prog, "meshes/monkey.obj") |> SceneObject
    cube = load_mesh(depth_prog, "meshes/cube.obj") |> SceneObject
    scene = Scene(camera, [monkey])
    cube_scene = Scene(camera, [monkey, cube])

    # Finally free OpenGL resources
    destroy_render_context(window)
end

# TODO fails to precompile because of the OpenGL stuff
# main()