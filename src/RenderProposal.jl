# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    RenderProposal
Decorator for a (proposal) `model`.
First, a new sample is generated from the `model` and then a rendering `μ` of the pose `t` & `r` is merged into the sample.
"""
struct RenderProposal{R<:Rotation,M,C<:RenderContext,S<:Scene} <: AbstractProposal
    # Probabilistic model
    model::M
    # Render related objects
    render_context::C
    scene::S
    object_id::Int
end

RenderProposal(proposal::P, render_context::C, scene::S, object_id::Int, ::Type{R}) where {P,C<:RenderContext,S<:Scene,R<:Rotation} = RenderProposal{R,P,C,S}(proposal, render_context, scene, object_id)

"""
    rand(rng, proposal, [dims...])
Generates a random sample from the decorated proposal.
Renders the translation `t` and rotational `r` component and returns a new sample with the rendering `μ`. 
"""
function Base.rand(rng::AbstractRNG, proposal::RenderProposal, dims...)
    sample = rand(rng, proposal.model, dims...)
    render(proposal, sample)
end

"""
    propose(rng, proposal, sample, [dims...])
Proposed a new sample from the decorated proposal.
Renders the translation `t` and rotational `r` component and returns a new sample with the rendering `μ`. 
"""
function propose(rng::AbstractRNG, proposal::RenderProposal, sample::Sample, dims...)
    proposed_sample = propose(rng, proposal.model, sample, dims...)
    render(proposal, proposed_sample)
end

"""
    render(proposal, sample)
Converts the translation `t` and rotational `r` component to poses and renders these.
Afterwards a new sample with the rendering `μ` is returned. 
"""
function render(proposal::RenderProposal{R}, sample::Sample) where {R}
    p = to_pose(variables(sample).t, variables(sample).r, R)
    # μ is only a view of the render_data. Storing it in every sample is cheap.
    μ = render(proposal.render_context, proposal.scene, proposal.object_id, p)
    merge(sample, (; μ=μ))
end

"""
    transition_probability(proposal, new_sample, prev_sample)
Rendering is determinstic, thus the probability is the one of the decorated proposal model.
"""
transition_probability(proposal::RenderProposal, new_sample, prev_sample) = transition_probability(proposal.model, new_sample, prev_sample)
