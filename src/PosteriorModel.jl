# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Random
using SciGL

#TODO Update doc, rename ImageModel.jl to ObservationModel.jl and move
"""
    ObservationModel
Provide a position `t`, orientation `r` and occlusion `o`.
Sets the pose of [object_id] to render depth of the scene.
This is used in the logdensityof with the pixel_dist to compare the expected and the measured images.
"""
struct ObservationModel{T<:AbstractArray,R<:AbstractArray,O<:AbstractArray,C<:RenderContext,P}
    t::T
    r::R
    o::O
    context::C
    # The result of rand would be a rendered scene with the pixel model applied to it
    pixel_dist::P
    # Normalize the image logdensity 
    normalize::Bool
end

array_to_translation(A::AbstractArray) = [Translation(t) for t in eachcol(A)]
array_to_rotation(A::AbstractArray, ::Type{T}=RotXYZ) where {T<:Rotation} = [T(r...) for r in eachcol(A)]

# Let broadcasting handle different sizes of t and r
pose_vector(t, r) = Pose.(t, r)

function render(context::RenderContext, position, orientation)
    for (idx, pose) in pose_vector(position, orientation)
        scene = @set context.scene.meshes[context.object_id] = pose
        activate_layer(context.framebuffer, idx)
        clear_buffers()
        draw(context.program, scene)
    end
    unsafe_copyto!(context.cache, context.framebuffer)
    context.cache
end
render(model::ObservationModel) = render(model.context, model.object_id, model.t, model.r)

function ImageModel(model::ObservationModel)
    μ = render(model)
    ImageModel(model.pixel_dist, μ, model.o, model.normalize)
end

"""
    rand(rng, model, dims...)
Generate a sample with variables (t=position, r=orientation, o=occlusion).
"""
function Base.rand(rng::AbstractRNG, model::ObservationModel, dims...)
    img_model = ImageModel(model)
    # TEST dims applies noise multiple times?
    rand(rng, img_model, dims...)
end

function DensityInterface.logdensityof(model::ObservationModel, x)
    img_model = ImageModel(model)
    logdensityof(img_model, x)
end

"""
    RenderContext
Stores all the static information required for rendering the object in the scene (everything except the latent variables like pose and occlusion).
Intended for offscreen rendering to the texture of the `framebuffer`.
Provide an `Array` or `CuArray` as `render_cache` for further processing in the models. 
"""
struct RenderContext{R<:Rotation,S<:Scene,F<:GLAbstraction.FrameBuffer,C<:AbstractArray,P<:GLAbstraction.AbstractProgram}
    r_type::Type{R}
    scene::S
    object_id::Int
    window::GLFW.Window
    framebuffer::F
    # Preallocate a CPU Array or GPU CuArray, this also avoids having to pass a device flag
    render_cache::C
    program::P
end

# TODO Keep it here?
PriorModel(t_model, r_model, o_model) = IndependentModel((t=t_model, r=r_model, o=o_model))

# TODO Parametric
struct PosteriorModel
    prior
    context
    # TODO duplicate with ObservationModel? Put it in the context and rename the context?
    pixel_dist::P
    # Normalize the image logdensity 
    normalize::Bool
end

function Random.rand(rng::AbstractRNG, posterior::PosteriorModel, dims...)
    # TODO should not be used during inference, instead rand from the prior.
    # TODO only for debugging
    sample = rand(rng, posterior.prior, dims...)
    t, r, o = variables()
end

# function Random.rand(rng::AbstractRNG, T::Type, prior::PoseDepthPrior)
#     tro_sample = rand(rng, T, prior.tro_model)
#     t = model_value(tro_sample, :t)
#     r = model_value(tro_sample, :o)
#     # TODO multiple hypotheses by implementing rand([rng=GLOBAL_RNG], [S], [dims...]) ? Probably best of both worlds: Render number on basis of vectorized measures in the tiled texture. For uniform interface it is probably best to include tiles in all Depth models. DepthModels.jl with AbstractDepthModel? tiles(::AbstractDepthModel)
#     μ = prior.render_fn(t, r)
#     μ_var = ModelVariable(μ, asℝ)
#     μ_sample = Sample((; μ=μ_var), -Inf)
#     merge(tro_sample, μ_sample)
# end
