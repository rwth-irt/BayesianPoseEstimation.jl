# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# TODO Use https://juliadynamics.github.io/DrWatson.jl/dev/save/ for reproducible experiments, make a new experiments project and dev src/MCMCDepth as described in their documentation
# TODO might want to convert this to a dict for DrWatson

"""
    Experiment
Data which might change from one experiment to another

* `gl_context` offscreen render context
* `scene` camera parameters or object mesh might change
* `prior_o` object association probability, e.g. segmentation mask or all the same
* `prior_t` estimated position of the object center e.g. via RFID or bounding box
* `depth_img` depth image of the observed scene
"""
struct Experiment
    gl_context::OffscreenContext
    scene::Scene
    prior_o::AbstractMatrix{Float32}
    prior_t::Vector{Float32}
    depth_image::AbstractMatrix{Float32}
end

"""
    Parameters
Monolith for storing the parameters.
Deliberately not strongly typed because the strongly typed structs are constructed in the Main script from this.

# Render context
* `width, height` Dimensions of the images.
* `min_depth, max_depth` Range limit of the sensor / region of interest
* `depth` z-dimension resembles the number of parallel renderings

# Observation Model
## Sensor Model
* `pixel_σ` Standard deviation of the sensor noise - typically overestimated.
* `pixel_θ` Expected value of the exponential distribution → Occlusion expected closer or further away.

## Object Association
* `association_σ` Standard deviation of the sensor noise used in the pixel association. Should be a magnitude larger than the `pixel_σ`.
* `prior_o` Constant mixture coefficient for the pixel to object association / probability that a pixel belongs to the object.
* `proposal_σ_o` Random walk proposals for the association

## Image Model
* `normalize_img` Normalize the likelihood of an image using the number of rendered pixels. Uses the ValidPixel wrapper.
* `n_normalization_samples` Number of samples to calculate the normalization constant from the expected number of visible pixels.

# Pose Model
* `mean_t` Mean of the RFID measurement
* `σ_t` Standard deviation of RFID measurement, assumes independent x,y,z components

# Proposal Model
* `proposal_t` Proposal model for the position (gibbs, independent, symmetric)
* `proposal_σ_t` Standard deviation of the random walk moves for the position
* `proposal_r` Proposal model for the orientation (gibbs, independent, symmetric)
* `proposal_σ_r` Standard deviation of the random walk moves for the Euler angle based orientation
* `proposal_σ_r_quat` Standard deviation of the random walk moves for the quaternions based orientation

# Inference
* `precision` Type of the floating point precision, typically Float32 (or Float64)
* `device` :CUDA or :CPU which is used in array_type, rng and device_rng.
* `seed` Seed of the rng
* `algorithm` Symbol of the inference algorithm
* `n_samples` Number of samples in the chain
* `n_burn_in` Number of samples before recording the chain
* `n_thinning` Record only every n_thinning sample to the chain
* `n_particles` For particle / multiple try algorithms
* `relative_ess` Relative effective sample size threshold ∈ (0,1)
"""
Base.@kwdef struct Parameters
    # Render context
    width = 100
    height = 100
    depth = 500
    min_depth = 0
    max_depth = 10

    # Depth pixel model
    pixel_σ = 0.01
    pixel_θ = 1
    mix_exponential = 0.8
    # Pixel association
    association_σ = 0.1
    proposal_σ_o = 0.01
    # Image Model
    normalize_img = true
    n_normalization_samples = 20_000
    normalization_constant = 15

    # Pose Model
    mean_t = [0.0, 0.0, 2.0]
    σ_t = fill(0.03, 3)
    # Proposal Model
    proposal_σ_t = fill(0.01, 3)
    proposal_σ_r = fill(0.1, 3)
    # Inference
    float_type = Float32
    device = :CUDA
    seed = 8418387917544508114
    n_steps = 3_000
    n_burn_in = 1_000
    n_thinning = 0
    n_particles = 100
    relative_ess = 0.8
end

# Automatically convert to correct precision
Base.getproperty(p::Parameters, s::Symbol) = getproperty(p, Val(s))
Base.getproperty(p::Parameters, ::Val{K}) where {K} = getfield(p, K)

"""
    default_rng(parameters)
Returns the seeded random number generator for the CPU.
"""
function Random.default_rng(p::Parameters)
    rng = Random.default_rng()
    Random.seed!(rng, p.seed)
    return rng
end

"""
    cuda_rng(parameters)
Returns the seeded random number generator for the CUDA device.
"""
function cuda_rng(p::Parameters)
    rng = CUDA.default_rng()
    Random.seed!(rng, p.seed)
    return rng
end


"""
    device_rng(parameters)
Returns the seeded matching random number generator for the parameters device field (:CUDA or :CPU).
"""
function device_rng(p::Parameters)
    if p.device === :CUDA
        cuda_rng(p)
    elseif p.device === :CPU
        Random.default_rng(p)
    else
        @warn "Unknown device: $(p.device), falling back to CPU"
        # CPU is fallback
        Random.default_rng(p)
    end
end

cpu_array(p::Parameters, dims...) = Array{p.float_type}(undef, dims...)

function device_array_type(p::Parameters)
    if p.device === :CUDA
        CuArray
    elseif p.device === :CPU
        Array
    else
        @warn "Unknown device: $(p.device), falling back to CPU"
        # CPU is fallback
        Array
    end
end
device_array(p::Parameters, dims...) = device_array_type(p){p.float_type}(undef, dims...)

"""
    Scene(gl_context, parameters)
Create a scene for inference given the parameters.
"""
function SciGL.Scene(gl_context, p::Parameters)
    object = upload_mesh(gl_context, p.mesh)
    camera = Camera(p.cv_camera)
    Scene(camera, [object])
end

Base.getproperty(p::Parameters, ::Val{:min_depth}) = p.float_type.(getfield(p, :min_depth))
Base.getproperty(p::Parameters, ::Val{:max_depth}) = p.float_type.(getfield(p, :max_depth))

Base.getproperty(p::Parameters, ::Val{:prior_o}) = p.float_type.(getfield(p, :prior_o))
Base.getproperty(p::Parameters, ::Val{:proposal_σ_o}) = p.float_type.(getfield(p, :proposal_σ_o))
Base.getproperty(p::Parameters, ::Val{:normalization_constant}) = p.float_type.(getfield(p, :normalization_constant))
Base.getproperty(p::Parameters, ::Val{:pixel_σ}) = p.float_type.(getfield(p, :pixel_σ))
Base.getproperty(p::Parameters, ::Val{:association_σ}) = p.float_type.(getfield(p, :association_σ))
Base.getproperty(p::Parameters, ::Val{:pixel_θ}) = p.float_type.(getfield(p, :pixel_θ))
Base.getproperty(p::Parameters, ::Val{:mix_exponential}) = p.float_type.(getfield(p, :mix_exponential))

Base.getproperty(p::Parameters, ::Val{:mean_t}) = p.float_type.(getfield(p, :mean_t))
Base.getproperty(p::Parameters, ::Val{:σ_t}) = p.float_type.(getfield(p, :σ_t))
Base.getproperty(p::Parameters, ::Val{:proposal_σ_t}) = p.float_type.(getfield(p, :proposal_σ_t))
Base.getproperty(p::Parameters, ::Val{:proposal_σ_r}) = p.float_type.(getfield(p, :proposal_σ_r))
Base.getproperty(p::Parameters, ::Val{:proposal_σ_r_quat}) = p.float_type.(getfield(p, :proposal_σ_r_quat))

Base.getproperty(p::Parameters, ::Val{:img_size}) = (getfield(p, :width), getfield(p, :height))
