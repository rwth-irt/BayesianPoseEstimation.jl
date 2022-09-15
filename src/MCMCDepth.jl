# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

__precompile__()

module MCMCDepth

# Common functions on Base types
include("Common.jl")
# Model primitives
include("Samples.jl")
include("FunctionManipulation.jl")
include("ModelInterface.jl")
include("Proposals.jl")
# Extensions
include("BijectorsExtensions.jl")
include("KernelDistributions.jl")
include("BroadcastedDistribution.jl")
include("QuaternionDistribution.jl")
# Inference / Sampling algorithms
include("MetropolisHastings.jl")
# TODO include("Gibbs.jl")
include("Visualization.jl")
# Plumbing together the depth image based pose estimator
include("Parameters.jl")
include("RenderContext.jl")
include("ObservationModel.jl")
include("AssociationModel.jl")
include("PosteriorModel.jl")
include("RenderProposal.jl")

# Common
export array_for_rng
export flatten
export map_intersect
export norm_dims, normalize_dims!, normalize_dims
export sum_and_dropdims
export to_rotation, to_translation, to_pose

# Samples
export Sample

export flatten
export log_prob
export merge
export names
export types
export unconstrained_state
export variables

# Distributions
export AbstractKernelDistribution
export measure_theory, kernel_distribution
export KernelBinaryMixture
export KernelCircularUniform
export KernelExponential
export KernelNormal
export KernelUniform
export BroadcastedDistribution
export QuaternionDistribution

# ModelInterface
export ComposedModel
export ConditionedModel
export IndependentModel
export RngModel

# Proposals
export GibbsProposal
export IndependentProposal
export Proposal
export SymmetricProposal

export propose
export transition_probability

# Parameters
export Parameters

export cpu_rng, cuda_rng, device_rng, cpu_array, device_array_type, device_array

# RenderContext
export RenderContext
export render

# Models
export PixelDistribution
export ObservationModel

export PixelAssociation
export ImageAssociation

export mix_normal_exponential, mix_normal_truncated_exponential

export image_association
export pixel_association
export pose_depth_model
export preprocess
export prior_depth_model
export random_walk_proposal

export PriorModel
export PosteriorModel
export RenderProposal

# MetropolisHastings
export MetropolisHastings

# Gibbs
export AnalyticGibbs
export Gibbs

# Visualization
export density_variable
export mean_image
export plot_depth_img, plot_prob_img
export polar_density_variable
export plot_variable
export polar_histogram_variable
export scatter_position

# Extensions and Reexports
using Reexport
@reexport import Rotations: QuatRotation, RotXYZ
@reexport import CoordinateTransformations: Translation
@reexport import SciGL: Scene

@reexport import DensityInterface: logdensityof
@reexport import Random: rand!

# Bijectors
@reexport import Bijectors: bijector, link, invlink, with_logabsdet_jacobian, transformed
export Circular
export is_identity

# Distributions
@reexport import Distributions: truncated

end # module
