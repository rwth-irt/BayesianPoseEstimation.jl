# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

module MCMCDepth

using Accessors
using CoordinateTransformations
using CUDA
using DensityInterface
using EllipsisNotation
using LogExpFunctions
using Logging
using Quaternions
using Random
using Reexport
using Rotations
using SciGL
using StaticArrays
using StatsBase

@reexport using AbstractMCMC
@reexport using KernelDistributions

# Common functions on Base & CUDA types
include("Common.jl")
# BayesNet
include("BayesNet/BayesNet.jl")
include("BayesNet/BroadcastedNode.jl")
include("BayesNet/DeterministicNode.jl")
include("BayesNet/ModifierNode.jl")
include("BayesNet/Sequentialized.jl")
include("BayesNet/SimpleNode.jl")
# Model primitives
include("Samples.jl")
include("FunctionManipulation.jl")
include("Proposals.jl")
include("PosteriorModel.jl")
# Inference / Sampling algorithms
include("Tempering.jl")

include("MetropolisHastings.jl")
include("ComposedSampler.jl")
include("Gibbs.jl")
include("MultipleTry.jl")
include("SequentialMonteCarlo.jl")

# Plumbing together the depth image based pose estimator
include("Visualization.jl")
include("Parameters.jl")
include("RenderContext.jl")
include("AssociationModel.jl")
include("Models.jl")

# Common
export array_for_rng
export flatten
export map_intersect
export norm_dims, normalize_dims!, normalize_dims
export sum_and_dropdims
export to_rotation, to_translation, to_pose

# BayesNet
export BroadcastedNode
export DeterministicNode
export ModifierNode
export SimpleNode

export evaluate
export parents
export prior
export sequentialize

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
export BroadcastedDistribution
export BroadcastedDistribution
export DiscreteBroadcastedDistribution

# Proposals
export additive_proposal
export independent_proposal
export quaternion_additive
export quaternion_symmetric
export symmetric_proposal

export propose
export transition_probability

# Parameters
export Parameters

export cpu_rng, cuda_rng, device_rng, cpu_array, device_array_type, device_array

# RenderContext
export RenderContext
export render

# PosteriorModel
export PosteriorModel

# Tempering
export ConstantSchedule
export ExponentialSchedule
export LinearSchedule

export increment_temperature

# Samplers
export ComposedSampler
export Gibbs
export MetropolisHastings
export MultipleTry

export BootstrapKernel
export ForwardProposalKernel
export MhKernel
export SequentialMonteCarlo
export smc_step

# Models
export ImageLikelihoodNormalizer
export ValidPixel

export expected_pixel_count
export image_association
export nonzero_pixels
export pixel_association
export pixel_explicit
export pixel_mixture
export pixel_normal
export pixel_tail
export render_fn
export smooth_mixture
export smooth_tail
export smooth_valid_mixture
export smooth_valid_tail
export valid_pixel_explicit
export pixel_valid_mixture
export valid_pixel_normal
export pixel_valid_tail

# Visualization
export density_variable
export histogram_variable
export mean_image
export plot_depth_img, plot_prob_img
export polar_density_variable
export plot_logprob
export plot_pose_chain
export plot_pose_density
export plot_variable
export polar_histogram_variable
export scatter_position
export sphere_density
export sphere_scatter

# Extensions and Reexports
@reexport import Quaternions: Quaternion
@reexport import Rotations: QuatRotation, RotXYZ
@reexport import CoordinateTransformations: Translation
@reexport import SciGL: Scale, Scene
@reexport import StatsBase: Weights



# Distributions
end # module
