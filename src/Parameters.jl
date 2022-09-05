# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using CUDA
using Random


# TODO Use https://juliadynamics.github.io/DrWatson.jl/dev/save/ for reproducible experiments, make a new experiments project and dev src/MCMCDepth as described in their documentation
# TODO might want to convert this to a dict for DrWatson

"""
    Parameters
Monolith for storing the parameters.
Deliberately not strongly typed because the strongly typed struct are constructed in the Main script from this.

# Scene Objects
* mesh_files: Meshes in the scene
* object_id: Index of the object to estimate the pose of in the scene

# Camera
* width, height: Dimensions of the image.
* depth: z-dimension resembles the number of parallel renderings
* f_x, f_y: Focal length of the OpenCV camera calibration
* c_x, c_y: Optical center of the OpenCV camera calibration
* min_depth, max_depth: Range limit of the sensor

# Observation Model
## Sensor Model
* pixel_dist: Symbol of the depth pixel distribution
* pixel_σ: Standard deviation of the sensor noise.
* pixel_θ: Expected value of the exponential distribution → Occlusion expected closer or further away.

## Object Association
* analytic_o: If true calculate the pixel association probability o analytically (Gibbs)
* association_is, association_not: Symbol of the distributions if a pixel belongs to the object or not
* static_o: Constant mixture coefficient for the pixel to object association / probability that a pixel belongs to the object.

## Image Model
* normalize_img: Normalize the likelihood of an image using the number of rendered pixels

# Pose Model
* rotation_type: Representation of rotations, e.g. :RotXYZ [x,y,z] or :QuatRotation [w,x,y,z] 
* mean_t: Mean of the RFID measurement
* σ_t: Standard deviation of RFID measurement, assumes independent x,y,z components
## TODO Different rotation models?

# Proposal Model
* proposal_t: Proposal model for the position (gibbs, independent, symmetric)
* proposal_σ_t: Standard deviation of the random walk moves for the position
* proposal_r: Proposal model for the orientation (gibbs, independent, symmetric)
* proposal_σ_r: Standard deviation of the random walk moves for the orientation

# Inference
* precision: Type of the floating point precision, typically Float32 (or Float64)
* device: :CUDA or :CPU which is used in Parameters.array_type and Parameters.rng.
* seed: Seed of the rng
* algorithm: Symbol of the inference algorithm
* n_samples: Number of samples in the chain
* n_burn_in: Number of samples before recording the chain
* n_thinning: Record only every n_thinning sample to the chain
"""
Base.@kwdef struct Parameters
    # Meshes
    mesh_files = ["meshes/monkey.obj"]
    object_id = 1
    # Camera
    width = 100
    height = 100
    depth = 500
    f_x = 120
    f_y = 120
    c_x = 50
    c_y = 50
    min_depth = 0.1
    max_depth = 2
    # Depth pixel model
    pixel_dist = :DepthNormalExponentialUniform
    pixel_σ = 0.1
    pixel_θ = 1.0
    mix_exponential = 0.8
    # Pixel association
    analytic_o = false
    association_is = :KernelNormal
    association_not = :KernelExponential
    static_o = fill(0.2, 100, 100)
    # Image Model
    normalize_img = true
    # Pose Model
    rotation_type = :RotXYZ
    mean_t = [0.0, 0.0, 2.0]
    σ_t = [0.05, 0.05, 0.05]
    # Proposal Model
    proposal_t = :SymmetricProposal
    proposal_σ_t = [0.05, 0.05, 0.05]
    proposal_r = :SymmetricProposal
    proposal_σ_r = [0.05, 0.05, 0.05]
    # Inference
    precision = Float32
    device = :CUDA
    seed = 8418387917544508114
    n_samples = 5000
    n_burn_in = 1000
    n_thinning = 2
end


# Automatically convert to correct precision
Base.getproperty(p::Parameters, s::Symbol) = getproperty(p, Val(s))
Base.getproperty(p::Parameters, ::Val{K}) where {K} = getfield(p, K)

function Base.getproperty(p::Parameters, ::Val{:rng})
    if p.device === :CUDA
        CUDA.default_rng()
    elseif p.device === :CPU
        Random.default_rng()
    end
end
Base.getproperty(p::Parameters, ::Val{:array_type}) = array_for_rng(p.rng)


Base.getproperty(p::Parameters, ::Val{:min_depth}) = p.precision.(getfield(p, :min_depth))
Base.getproperty(p::Parameters, ::Val{:max_depth}) = p.precision.(getfield(p, :max_depth))

Base.getproperty(p::Parameters, ::Val{:static_o}) = p.precision.(getfield(p, :static_o))
Base.getproperty(p::Parameters, ::Val{:pixel_σ}) = p.precision.(getfield(p, :pixel_σ))
Base.getproperty(p::Parameters, ::Val{:pixel_θ}) = p.precision.(getfield(p, :pixel_θ))
Base.getproperty(p::Parameters, ::Val{:mix_exponential}) = p.precision.(getfield(p, :mix_exponential))

Base.getproperty(p::Parameters, ::Val{:mean_t}) = p.precision.(getfield(p, :mean_t))
Base.getproperty(p::Parameters, ::Val{:σ_t}) = p.precision.(getfield(p, :σ_t))
Base.getproperty(p::Parameters, ::Val{:proposal_σ_t}) = p.precision.(getfield(p, :proposal_σ_t))
Base.getproperty(p::Parameters, ::Val{:proposal_σ_r}) = p.precision.(getfield(p, :proposal_σ_r))

function Base.getproperty(p::Parameters, ::Val{:rotation_type})
    r = getfield(p, :rotation_type)
    if r isa Type
        Base.typename(r).wrapper{p.precision}
    elseif r isa Symbol
        eval(r){p.precision}
    end
end