# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

module MCMCDepth

# lib includes
include("IsConstrainedTrait.jl")
include("Samples.jl")
include("Proposals.jl")
include("MetropolisHastings.jl")
include("Visualization.jl")
include("CircularTransform.jl")

# Samples
export ConstrainedSample
export Sample

export log_probability
export merge
export state
export unconstrained_state

# Proposals
export GibbsProposal
export IndependentProposal
export Proposal
export SymmetricProposal

export propose
export transition_probability

# Visualization
export colorize_depth

# CircularTransform
export as○, as_circular

end # module
