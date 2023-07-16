# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

module MCMCDepth

using Accessors
using Bijectors
using CoordinateTransformations
using CUDA
using DensityInterface
using EllipsisNotation
using LinearAlgebra: isposdef
using LogExpFunctions
using Logging
using PoseErrors
using ProgressLogging
using Quaternions
using Random
using Reexport
# Avoid loading ⊕ and ⊖
using Rotations: Rotation, QuatRotation, RotMatrix3
using SciGL
using StatsBase

@reexport using AbstractMCMC
@reexport using BayesNet
@reexport using KernelDistributions

# Common functions on Base & CUDA types
include("Common.jl")
# Model primitives
include("Samples.jl")
include("FunctionManipulation.jl")
include("PosteriorModel.jl")
include("Proposals.jl")
# Inference / Sampling algorithms
include("Tempering.jl")

include("MetropolisHastings.jl")
include("ComposedSampler.jl")
include("Gibbs.jl")
include("MultipleTry.jl")
include("SequentialMonteCarlo.jl")

# Plumbing together the depth image based pose estimator
include("Parameters.jl")
include("Crop.jl")
include("Visualization.jl")
include("RenderContext.jl")
include("Models.jl")
include("ExperimentModels.jl")
include("ExperimentSamplers.jl")

# Datasets
include("BOP.jl")

# Common
export array_for_rng
export map_intersect
export norm_dims, normalize_dims!, normalize_dims
export sum_and_dropdims
export to_rotation, to_translation, to_pose

# Samples
export Sample
export loglikelihood
export logprobability
export merge
export names
export to_model_domain
export to_unconstrained_domain
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
export Experiment
export Parameters

export cuda_rng, device_rng, cpu_array, device_array_type, device_array

# RenderContext
export render_context
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

export AdaptiveKernel
export BootstrapKernel
export ForwardProposalKernel
export MhKernel
export SequentialMonteCarlo
export smc_step

# Models
export ImageLikelihoodNormalizer
export ValidPixel

export expected_pixel_count
export nonzero_pixels
export pixel_association_fn
export pixel_explicit
export pixel_mixture
export pixel_normal
export pixel_tail
export render_fn
export smooth_association_fn
export smooth_mixture
export smooth_tail
export smooth_valid_mixture
export smooth_valid_tail
export pixel_valid_explicit
export pixel_valid_mixture
export pixel_valid_normal
export pixel_valid_tail

# Visualization
export density_variable
export histogram_variable
export mean_image
export plot_best_pose
export plot_depth_img
export plot_depth_ontop, plot_scene_ontop
export plot_logevidence
export plot_logprob
export plot_pose_chain
export plot_pose_density
export plot_prob_img
export plot_variable
export polar_density_variable
export polar_histogram_variable
export scatter_position
export sphere_density
export sphere_scatter

# Experiment models & samplers
export association_posterior
export point_prior
export simple_posterior
export smooth_posterior

export mh_local_sampler
export mh_sampler
export mtm_local_sampler
export mtm_sampler

export smc_bootstrap
export smc_forward
export smc_inference
export smc_mh

# Data loading an preprocessing
export crop_camera
export load_color_image
export load_depth_image
export load_mask_image
export scene_dataframe

# Extensions and Reexports
@reexport import Quaternions: Quaternion
@reexport import Rotations: QuatRotation, RotXYZ
@reexport import CoordinateTransformations: Translation
@reexport import SciGL: OffscreenContext, Scale, Scene, destroy_context
@reexport import StatsBase: Weights

# Distributions
end # module
