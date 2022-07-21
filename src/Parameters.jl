# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using LinearAlgebra

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
* pixel_dist: Name of the depth pixel distribution
* pixel_σ: Standard deviation of the sensor noise.
* pixel_θ: Expected value of the exponential distribution → Occlusion expected closer or further away.

## Object Association
* analytic_o: If true calculate the pixel association probability o analytically (Gibbs)
* association_is, association_not: Names of the distributions if a pixel belongs to the object or not
* const_o: Constant mixture coefficient for the pixel to object association / probability that a pixel belongs to the object.

## Image Model
* normalize_img: Normalize the likelihood of an image using the number of rendered pixels

# Pose Model
* rotation_type: Representation of rotations, e.g. RotXYZ [x,y,z] or QuatRotation [w,x,y,z] 

"""
Base.@kwdef struct Parameters
    # Meshes
    mesh_files = ["meshes/monkey.obj"]
    object_id = 1
    # Camera
    width = 100
    height = 100
    depth = 1
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
    mix_exponential::T = T(0.8)
    # Pixel association
    analytic_o = false
    association_is = :KernelNormal
    association_not = :KernelExponential
    # Image Model
    normalize_img = true
    # Pose Model
    rotation_type = RotXYZ
end

"""
  PriorParameters
# Arguments
- `<...>_t` describes the prior `MvNormal` estimate from the RFID sensor and the pixel association.
- `o` describes the prior of each pixel belonging to the object of interest.
"""
Base.@kwdef struct PriorParameters
    mean_t::Vector{Float32} = [0.0, 0.0, 2.0]
    σ_t::Vector{Float32} = [0.05, 0.05, 0.05]
    cov_t::Matrix{Float32} = Diagonal(σ_t)
    o::Matrix{Float32} = fill(0.2, 100, 100)
end

"""
  RandomWalkParameters
Parameters of the zero centered normal distributions of a random walk proposal.
"""
Base.@kwdef struct RandomWalkParameters
    σ_t::Vector{Float32} = [0.05, 0.05, 0.05]
    cov_t::Matrix{Float32} = Diagonal(σ_t)
    σ_r::Vector{Float32} = [0.05, 0.05, 0.05]
    cov_r::Matrix{Float32} = Diagonal(σ_r)
    σ_o::Float32 = 0.05
    width::Int64 = 100
    height::Int64 = 100
end
