# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

module MCMCDepth

# lib includes
include("Variables.jl")
include("Samples.jl")
include("Proposals.jl")
include("Parameters.jl")
include("FunctionManipulation.jl")
include("PriorModel.jl")
# TODO re-add or split?
# include("Models.jl")
include("MetropolisHastings.jl")
include("Gibbs.jl")
include("Visualization.jl")
# Extensions
include("TransformVariablesExtensions.jl")
include("KernelDistributions.jl")
include("KernelDistributionsVariables.jl")
include("VectorizedDistributions.jl")
include("VectorizedDistributionsVariables.jl")
include("Tiles.jl")
# Inference
include("Main.jl")

# Variables
export ModelVariable
export SampleVariable
export model_value, model_value_and_logjac
export raw_value

# Samples
export Sample

export flatten
export logp
export merge
export unconstrained_state
export vars

# Proposals
export AnalyticProposal
export GibbsProposal
export IndependentProposal
export Proposal
export SymmetricProposal

export propose
export transition_probability

# Parameters
export DepthImageParameters
export PriorParameters
export RandomWalkParameters

# PriorModel
export IndependentPrior

# Models
export DepthExponential
export DepthExponentialUniform
export DepthImageMeasure
export DepthNormal
export DepthNormalExponential
export DepthNormalExponentialUniform
export DepthNormalUniform
export DepthUniform
export WrappedModel

export image_association
export pixel_association
export pose_depth_model
export preprocess
export prior_depth_model
export random_walk_proposal

# MetropolisHastings
export MetropolisHastings

# Gibbs
export AnalyticGibbs
export Gibbs

# Visualization
export colorize_depth
export colorize_probability
export density_variable
export mean_image
export plot_variable
export polar_density_variable
export polar_histogram_variable
export scatter_position

# Kernel distributions
export AbstractKernelDistribution
export measure_theory, kernel_distribution
export KernelBinaryMixture
export KernelCircularUniform
export KernelExponential
export KernelNormal
export KernelUniform

# Vectorized distributions
export ProductDistribution
export VectorizedDistribution
export to_cpu, to_gpu

# Extensions
using Reexport

@reexport import DensityInterface: logdensityof
@reexport import TransformVariables: as
export as○, as_circular
@reexport import Random: rand!

# Main script
export destroy_render_context
export init_render_context
export main
export render_to_cpu
export render_pose
export render_pose!

end # module
