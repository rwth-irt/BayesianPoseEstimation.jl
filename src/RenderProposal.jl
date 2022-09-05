# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# TODO Doc
# TEST
struct RenderProposal{R<:Rotation,P,C<:RenderContext,S<:Scene}
    # Probabilistic model
    proposal::P
    # Render related objects
    render_context::C
    scene::S
    object_id::Int
end

RenderProposal(proposal::P, render_context::C, scene::S, object_id::Int, ::Type{R}) where {P,C<:RenderContext,S<:Scene,R<:Rotation} = RenderProposal{R,P,C,S}(proposal, render_context, scene, object_id)

# TODO hard-coded variable names? t, o, μ
function propose(rng::AbstractRNG, proposal::RenderProposal{R}, dims...) where {R}
    sample = propose(rng, proposal.proposal, dims...)
    p = to_pose(variables(sample).t, variables(sample).r, R)
    # μ is only a view of the render_data. Storing it in every sample is cheap.
    μ = render(proposal.render_context, proposal.scene, proposal.object_id, p)
    merge(sample, (; μ=μ))
end

transition_probability(proposal::RenderProposal, new_sample, prev_sample) = transition_probability(proposal.proposal, new_sample, prev_sample)
