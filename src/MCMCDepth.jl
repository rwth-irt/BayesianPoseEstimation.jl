# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

module MCMCDepth

# lib includes
include("Samples.jl")
include("Proposals.jl")
include("MetropolisHastings.jl")
include("Visualization.jl")
# Extensions
include("TransformVariablesExtensions.jl")
include("MeasureTheoryExtensions.jl")

# Samples
export Sample

export log_probability
export merge
export raw_state
export state
export unconstrained_state

# Proposals
export GibbsProposal
export IndependentProposal
export Proposal
export SymmetricProposal

export propose
export transition_probability

# MetropolisHastings
export PosteriorModel
export MetropolisHastings

# Visualization
export colorize_depth

# Extensions
export as○, as_circular
export CircularUniform
export BinaryMixture
export MixtureMeasure
export UniformInterval

end # module
