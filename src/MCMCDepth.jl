# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

module MCMCDepth

# lib includes
include("Samples.jl")
include("Proposals.jl")
include("Models.jl")
include("MetropolisHastings.jl")
include("Gibbs.jl")
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
export AnalyticProposal
export GibbsProposal
export IndependentProposal
export Proposal
export SymmetricProposal

export propose
export transition_probability

# Model
export PosteriorModel

# MetropolisHastings
export MetropolisHastings

# Gibbs
export Gibbs

# Visualization
export colorize_depth
export colorize_probability

# Extensions
export as○, as_circular
export CircularUniform
export BinaryMixture
export MixtureMeasure
export UniformInterval

end # module
