# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# TODO move usings
using DensityInterface
using Random
using Rotations
using SciGL

# TODO is this the place to get specific about the variable naming? t,r,o,μ,z,...?
# TODO doc

# TODO Keep it here?
PriorModel(t_model, r_model, o_model) = IndependentModel((t=t_model, r=r_model, o=o_model))

# TODO Parametric
struct PosteriorModel
    # Render related objects
    render_context
    scene
    object_id
    rotation_type#::T
    # Probabilistic models
    prior_model
    # Expected to be f(μ, o)
    observation_model
end

function Base.rand(rng::AbstractRNG, model::PosteriorModel, dims::Integer...)
    # TODO should not be used during inference, instead rand from the prior.
    prior_sample = rand(rng, model.prior_model, dims...)
    render_sample = render(model.render_context, model.scene, model.object_id, model.rotation_type, prior_sample)
    observation_instance = model.observation_model(render_sample.μ, render_sample.o)
    z = rand(rng, observation_instance)
    merge(render_sample, (; z=z))
end

function render(render_context::RenderContext, scene::Scene, object_id::Integer, rotation_type::Type, sample::Sample)
    p = to_pose(sample.t, sample.r, rotation_type)
    μ = render(render_context, scene, object_id, p)
    # μ is only a view of the render_data. Storing it in every sample is cheap.
    merge(sample, (; μ=μ))
end

# DensityInterface
@inline DensityKind(::PosteriorModel) = HasDensity()
function DensityInterface.logdensityof(model::PosteriorModel, sample)
    ℓ_prior = logdensityof(model.prior_model, sample)
    observation_instance = model.observation_model(render_sample.μ, render_sample.o)
    ℓ_likelihood = logdensityof(observation_instance, sample.z)
    ℓ_prior + ℓ_likelihood
end


# TODO Parametric
struct RenderProposal#{T<:Rotation}
    # Render related objects
    render_context
    scene
    object_id
    rotation_type#::T
    # Probabilistic model
    proposal
end

# TODO hard-coded variable names? t, o, μ
function propose(rng::AbstractRNG, proposal::RenderProposal, dims...)
    sample = propose(rng, proposal.proposal, dims...)
    p = to_pose(sample.t, sample.r, proposal.rotation_type)
    # μ is only a view of the render_data. Storing it in every sample is cheap.
    μ = render(proposal.render_context, proposal.scene, object_id, p)
    merge(sample, (; μ=μ))
end

transition_probability(proposal::RenderProposal, new_sample, prev_sample) = transition_probability(proposal.proposal, new_sample, prev_sample)


# TODO μ is used by this and the association model → outsource to avoid rendering twice
"""
    ObservationModel(render_context, scene, object_id, rotation_type, normalize_img, pixel_dist, t, r, o)
Generate an ObservationModel by rendering the expected depth `μ` for the provided scene.
Sets the pose of `object_id` to the position(s) `t` and orientation(s) `r`.
"""
function ObservationModel(render_context::RenderContext, scene::Scene, object_id::Integer, rotation_type::Type, normalize_img::Bool, pixel_dist, t::AbstractArray, r::AbstractArray, o::AbstractArray)
    p = to_pose(t, r, rotation_type)
    μ = render(render_context, scene, object_id, p)
    ObservationModel(normalize_img, pixel_dist, μ, o)
end

# TODO assembling the model from params should probably happen on a higher level by partially eval and applying the parameters.
"""
    ObservationModel(parameters, render_context, scene, t, r, o)
Convenience constructor which extracts the `object_id`, `normalize_img`, `rotation_type` and `pixel_dist` from the `parameters` struct.
Note that `rotation_type` and `pixel_dist` are expected as Symbols which get evaled at runtime. 
"""
ObservationModel(parameters::Parameters, render_context::RenderContext, scene::Scene, t::AbstractArray, r::AbstractArray, o::AbstractArray) = ObservationModel(render_context, scene, parameters.object_id, eval(parameters.rotation_type), parameters.normalize_img, eval(parameters.pixel_dist), t, r, o)

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
