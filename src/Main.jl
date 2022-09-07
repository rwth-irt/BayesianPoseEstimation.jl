# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

using CUDA
using Random

# TODO I think this really drives up the precompilation time

# TODO Do I want to pass pixel_dist as parameter or implement them in a PixelDistributions.jl and eval them from Parameters
# https://bkamins.github.io/julialang/2022/07/15/main.html
"""
    main()
The main inference script which will cause compilation instead of running in the global scope.
"""
function main( parameters::Parameters, pixel_dist)

    # RNGs
    dev_rng = device_rng(parameters)
    Random.seed!(dev_rng, parameters.seed)
    cpu_rng = Random.default_rng()
    Random.seed!(cpu_rng, parameters.seed)

    # Device
    if parameters.device === :CUDA
        CUDA.allowscalar(false)
    end
    # Allows us to enforce the pose models to run on the CPU
    rng_model(rng, model) = RngModel(rng, model)
    cpu_model = rng_model | cpu_rng

    # Render context
    render_context = RenderContext(parameters.width, parameters.height, parameters.depth, device_array_type(parameters))
    scene = Scene(parameters, render_context)

    # Pose models on CPU since CUDA cannot call OpenGL render functions
    t_model = BroadcastedDistribution(KernelNormal, parameters.mean_t, parameters.σ_t) |> cpu_model
    # TODO allow different rotation Distributions
    # TODO This is hacky, any clean implementation which avoids broadcasting over fake parameters?
    circular_uniform(::Any) = KernelCircularUniform()
    r_model = BroadcastedDistribution(circular_uniform, Array{parameters.precision}(undef, 3)) |> cpu_model

    # Pixel association on device
    o_model = BroadcastedDistribution(uniform, device_array(parameters)(parameters.width, parameters.height))


    # Assemble PosteriorModel
    prior_model = PriorModel(t_model, r_model, o_model)
    observation_model(μ, o) = ObservationModel(parameters.normalize_img, pixel_dist, μ, o)
    posterior_model = PosteriorModel(prior_model, observation_model, render_context, scene, parameters.object_id, parameters.rotation_type)

    # Proposals
    # TODO Transformed distributions for SymmetricProposal

    # TODO implement getter and test which eval the model symbols


    # Finally free OpenGL resources
    # destroy_render_context(render_context.window)
end

function test(par::Parameters)
    # WARN not type stable when conditioning on params
    # assoc_is(μ) = KernelNormal(μ, par.pixel_σ)
    # This is since par is evaluated before creating the named function
    # local_σ = par.pixel_σ
    # ass_is(μ) = KernelNormal(μ, local_σ)
    reversed_normal(σ, μ) = KernelNormal(μ, σ)
    assoc_is = reversed_normal | par.pixel_σ
    d = BroadcastedDistribution(assoc_is, 1.0f0)
    logdensityof(d, CUDA.fill(2.0f0, 10))
end

function test_assoc_is(par::Parameters, μ, x)
    # Not type stable
    fn = par.association_is
    # Type stable because of function barrier
    logdensityof(fn(μ), x)
end