# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

__precompile__()

module MCMCDepth

# Common functions on Base & CUDA types
include("Hijack.jl")
include("Common.jl")
# Model primitives
include("Samples.jl")
include("FunctionManipulation.jl")
include("ModelInterface.jl")
include("Proposals.jl")
include("PosteriorModel.jl")
# Extensions
include("BijectorsExtensions.jl")
# Distributions
include("KernelDistributions.jl")
include("BroadcastedDistribution.jl")
include("QuaternionDistribution.jl")
# BayesNet
include("BayesNet/BayesNet.jl")
include("BayesNet/BroadcastedNode.jl")
include("BayesNet/ModifierNode.jl")
include("BayesNet/Sequentialized.jl")
include("BayesNet/SimpleNode.jl")
# Inference / Sampling algorithms
include("MetropolisHastings.jl")
include("ComposedSampler.jl")
# TODO include("Gibbs.jl")
include("Visualization.jl")
# Plumbing together the depth image based pose estimator
include("Parameters.jl")
include("RenderContext.jl")
include("ObservationModel.jl")
include("AssociationModel.jl")
include("RenderModel.jl")
include("Inference.jl")

# Common
export array_for_rng
export flatten
export map_intersect
export norm_dims, normalize_dims!, normalize_dims
export sum_and_dropdims
export to_rotation, to_translation, to_pose

# Samples
export Sample

export logprob
export merge
export names
export to_model_domain
export to_unconstrained_domain
export transform
export types
export variables

# Distributions
export AbstractKernelDistribution
export measure_theory, kernel_distribution
export KernelBinaryMixture
export KernelCircularUniform
export KernelExponential
export KernelNormal
export KernelUniform

export ProductBroadcastedDistribution
export BroadcastedDistribution
export DiscreteBroadcastedDistribution

# BayesNet
export BroadcastedNode
export ModifierNode
export SimpleNode
export sequentialize

# ModelInterface
export ComposedModel
export ConditionedModel
export IndependentModel
export RngModel

# Proposals
export IndependentProposal
export Proposal
export SymmetricProposal

export propose
export transition_probability

# Quaternions
export QuaternionDistribution
export QuaternionPerturbation
export QuaternionProposal

# Parameters
export Parameters

export cpu_rng, cuda_rng, device_rng, cpu_array, device_array_type, device_array

# RenderContext
export RenderContext
export render

# Models
export ObservationModel
export ValidPixel

export PixelAssociation
export ImageAssociation

export mix_normal_exponential, mix_normal_truncated_exponential

export image_association
export nonzero_pixels
export pixel_association
export pose_depth_model
export preprocess
export prior_depth_model
export random_walk_proposal

export PriorModel
export PosteriorModel
export RenderModel

# Samplers
export ComposedSampler
export Gibbs
export MetropolisHastings

# Inference
export expected_pixel_count
export pixel_explicit
export pixel_mixture
export pixel_tail

# Visualization
export density_variable
export histogram_variable
export mean_image
export plot_depth_img, plot_prob_img
export polar_density_variable
export plot_logprob
export plot_variable
export polar_histogram_variable
export scatter_position
export sphere_density
export sphere_scatter

# Extensions and Reexports
using Reexport
@reexport import Quaternions: Quaternion
@reexport import Rotations: QuatRotation, RotXYZ
@reexport import CoordinateTransformations: Translation
@reexport import SciGL: Scale, Scene

@reexport import DensityInterface: logdensityof
@reexport import Random: rand!
@reexport import StatsBase: Weights

# Bijectors
@reexport import Bijectors: bijector, inverse, link, invlink, with_logabsdet_jacobian, transformed
export BroadcastedBijector
export Circular
export ZeroIdentity
export is_identity

# Distributions
@reexport import Distributions: truncated

end # module
