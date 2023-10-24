# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using Quaternions

"""
    Experiment
Data which might change from one experiment to another

* `gl_context` offscreen render context
* `scene` camera parameters or object mesh might change
* `prior_o` object association probability, e.g. segmentation mask or all the same
* `prior_t` estimated position of the object center e.g. via RFID or bounding box
* `depth_img` depth image of the observed scene
"""
struct Experiment{T}
    gl_context::OffscreenContext
    scene::Scene
    prior_o
    prior_t::Vector{T}
    prior_r::Quaternion{T}
    depth_image

    """
    Experiment(gl_context, scene, prior_o, prior_t, prior_r, depth_image)
        Preprocesses the data before setting up the experiment:
        All pixels with depth 0 are replaced with infinity so all probability densities except the long tail are zero.

        Automatically transfers `prior_o` `depth_image` to the device of the `gl_context`.
    """
    function Experiment(gl_context::OffscreenContext, scene, prior_o, prior_t, prior_r, depth_image::AbstractMatrix{T}) where {T}
        device_img = to_device(gl_context, depth_image)
        # can also be a scalar
        if prior_o isa AbstractArray
            prior_o = to_device(gl_context, prior_o)
        end
        indices = device_img .<= 0
        device_img[indices] .= typemax(T)
        new{T}(gl_context, scene, prior_o, prior_t, prior_r, device_img)
    end
end

"""
    Experiment(gl_context, scene, prior_o, prior_t, depth_image)
Defaults the prior_r to the identity quaternion.
"""
Experiment(gl_context::OffscreenContext, scene, prior_o, prior_t, depth_image::AbstractMatrix) = Experiment(gl_context, scene, prior_o, prior_t, one(Quaternion), depth_image)

"""
    Experiment(experiment, depth_image)
Return a new experiment where the depth image is replaced.
"""
Experiment(ex::Experiment, depth_img::AbstractMatrix) = Experiment(ex.gl_context, ex.scene, ex.prior_o, ex.prior_t, ex.prior_r, depth_img)

"""
    Parameters
Monolith for storing the parameters.
Deliberately not strongly typed because the strongly typed structs are constructed in the Main script from this.

# Render context
* `width, height` Dimensions of the images.
* `min_depth, max_depth` Region of interest / range limit of the sensor
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
* `c_reg` Regularization constant for the image likelihood regularization.

# Pose Model
* `σ_t` Standard deviation of RFID measurement, assumes independent x,y,z components

# Proposal Model
* `proposal_t` Proposal model for the position (gibbs, independent, symmetric)
* `proposal_σ_t` Standard deviation of the random walk moves for the position
* `proposal_r` Proposal model for the orientation (gibbs, independent, symmetric)
* `proposal_σ_r` Standard deviation of the random walk moves for the Euler angle based orientation
* `proposal_σ_r_quat` Standard deviation of the random walk moves for the quaternions based orientation

# Inference
* `float_type` Type of the floating point precision, typically Float32 (or Float64)
* `device` :CUDA or :CPU which is used in array_type, rng and device_rng.
* `seed` Seed of the rng
* `algorithm` Symbol of the inference algorithm
* `n_samples` Number of samples in the chain
* `time_budget` Inference time budget in seconds - calc n_samples via benchmark
* `n_burn_in` Number of samples before recording the chain
* `n_thinning` Record only every n_thinning sample to the chain
* `n_particles` For particle / multiple try algorithms
* `relative_ess` Relative effective sample size threshold ∈ (0,1)
* `w_t_sym`, `w_r_sym` Weight of using a local move proposal for t and r.
* `w_t_ind`, `w_r_ind` Weight of using an independent move proposal for t and r.
"""
Base.@kwdef struct Parameters
    # Render context
    width = 25
    height = 25
    depth = 100
    # Most BOP datasets and the Zivid One+ fall in this range
    min_depth = 0.5
    max_depth = 1.5

    # Depth pixel model
    pixel_σ = 0.01
    pixel_θ = 1
    # Pixel association
    association_σ = 0.01
    proposal_σ_o = 0.01
    # Image Model
    c_reg = 50

    # Pose Model
    σ_t = fill(0.03, 3)
    # Association model
    o_mask_is = 0.7
    o_mask_not = 0.3
    # Proposal Model
    proposal_σ_t = fill(0.01, 3)
    proposal_σ_r = fill(0.1, 3)
    velocity_decay = 0.9

    # Inference
    float_type = Float32
    device = :CUDA
    seed = 8418387917544508114
    n_steps = 3_000
    time_budget = 0.5
    n_burn_in = 0
    n_thinning = 0
    n_particles = 100
    relative_ess = 0.5
    w_r_ind = 0.1
    w_t_ind = 0.1
    w_r_sym = 0.9
    w_t_sym = 0.9
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
    cpu_rng(parameters)
Returns the seeded random number generator for the CPU.
"""
host_rng(p::Parameters) = Random.default_rng(p)

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

Base.getproperty(p::Parameters, ::Val{:min_depth}) = p.float_type.(getfield(p, :min_depth))
Base.getproperty(p::Parameters, ::Val{:max_depth}) = p.float_type.(getfield(p, :max_depth))

Base.getproperty(p::Parameters, ::Val{:prior_o}) = p.float_type.(getfield(p, :prior_o))
Base.getproperty(p::Parameters, ::Val{:o_mask_is}) = p.float_type.(getfield(p, :o_mask_is))
Base.getproperty(p::Parameters, ::Val{:o_mask_not}) = p.float_type.(getfield(p, :o_mask_not))
Base.getproperty(p::Parameters, ::Val{:proposal_σ_o}) = p.float_type.(getfield(p, :proposal_σ_o))

Base.getproperty(p::Parameters, ::Val{:c_reg}) = p.float_type.(getfield(p, :c_reg))
Base.getproperty(p::Parameters, ::Val{:pixel_σ}) = p.float_type.(getfield(p, :pixel_σ))
Base.getproperty(p::Parameters, ::Val{:association_σ}) = p.float_type.(getfield(p, :association_σ))
Base.getproperty(p::Parameters, ::Val{:pixel_θ}) = p.float_type.(getfield(p, :pixel_θ))

Base.getproperty(p::Parameters, ::Val{:σ_t}) = p.float_type.(getfield(p, :σ_t))
Base.getproperty(p::Parameters, ::Val{:proposal_σ_t}) = p.float_type.(getfield(p, :proposal_σ_t))
Base.getproperty(p::Parameters, ::Val{:proposal_σ_r}) = p.float_type.(getfield(p, :proposal_σ_r))
Base.getproperty(p::Parameters, ::Val{:proposal_σ_r_quat}) = p.float_type.(getfield(p, :proposal_σ_r_quat))
Base.getproperty(p::Parameters, ::Val{:w_t_sym}) = p.float_type.(getfield(p, :w_t_sym))
Base.getproperty(p::Parameters, ::Val{:w_r_sym}) = p.float_type.(getfield(p, :w_r_sym))
Base.getproperty(p::Parameters, ::Val{:w_t_ind}) = p.float_type.(getfield(p, :w_t_ind))
Base.getproperty(p::Parameters, ::Val{:w_r_ind}) = p.float_type.(getfield(p, :w_r_ind))

Base.getproperty(p::Parameters, ::Val{:img_size}) = (getfield(p, :width), getfield(p, :height))
