# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

"""
Run different samplers on fake data to benchmark the performance    Only the first scene of each dataset is evaluated because of the computation time.
    WARN: Results vary based on sampler configuration
    NOTE: Inference time grows linearly with n_hypotheses = n_particles * n_steps
    NOTE: smc_bootstrap & smc_forward mainly benefit from n_particles not n_steps
"""

using DrWatson
@quickactivate("MCMCDepth")

using Accessors
using BenchmarkTools
using CUDA
using DataFrames
using MCMCDepth
using Random
using SciGL

using Logging
using ProgressLogging
using TerminalLoggers
global_logger(TerminalLogger(right_justify=120))

import CairoMakie as MK

# Experiment setup
img_sizes = [25, 50, 100]
for img_size in img_sizes
    experiment_name = "inference_time_$img_size"
    sampler = [:mh_sampler, :mtm_sampler, :smc_mh]
    configs = dict_list(@dict sampler)

    # Context
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1
    CUDA.allowscalar(false)
    parameters = Parameters()
    @reset parameters.width = img_size
    @reset parameters.height = img_size
    @reset parameters.depth = 500
    @reset parameters.device = :CUDA
    cpu_rng = Random.default_rng(parameters)
    dev_rng = device_rng(parameters)
    gl_context = render_context(parameters)

    # Fake observation
    cube_path = joinpath(dirname(pathof(SciGL)), "..", "examples", "meshes", "cube.obj")
    mesh = upload_mesh(gl_context, cube_path)

    f_x = 1.2 * parameters.width
    f_y = f_x
    c_x = 0.5 * parameters.width
    c_y = 0.5 * parameters.height
    camera = CvCamera(parameters.width, parameters.height, f_x, f_y, c_x, c_y; near=parameters.min_depth, far=parameters.max_depth) |> Camera

    scene = Scene(camera, [mesh])
    pose = Pose([0.0f0, 0.0f0, 2.5f0], QuatRotation{Float32}(0.8535534, 0.1464466, 0.3535534, 0.3535534))
    depth_img = render(gl_context, scene, pose) |> copy
    mask_img = (depth_img .> 0) .|> Float32

    # Probabilistic model
    experiment = Experiment(gl_context, Scene(camera, [mesh]), mask_img, pose.translation.translation, depth_img)
    prior = point_prior(parameters, experiment, cpu_rng)
    posterior = simple_posterior(parameters, experiment, prior, dev_rng)

    function run_experiment(posterior, parameters, config)
        cpu_rng = Random.default_rng(parameters)
        @unpack sampler = config

        n_particles = [2, 10:10:500...]
        trials = Vector{BenchmarkTools.Trial}(undef, length(n_particles))
        @progress "$sampler" for (idx, particles) in enumerate(n_particles)
            @reset parameters.n_particles = particles
            # Sampler
            sampler_eval = eval(sampler)(cpu_rng, parameters, posterior)
            _, state = AbstractMCMC.step(cpu_rng, posterior, sampler_eval)
            # Interpolate local variables into the benchmark expression
            trials[idx] = @benchmark AbstractMCMC.step($cpu_rng, $posterior, $sampler_eval, $state)
        end
        @strdict parameters n_particles trials
    end

    # Run DrWatson
    result_dir = datadir("exp_raw", experiment_name)
    run_closure = run_experiment | (posterior, parameters)
    @progress "inference time for image size $img_size" for config in configs
        @produce_or_load(run_closure, config, result_dir; filename=my_savename)
    end
    destroy_context(gl_context)

    # Visualize and save plot
    diss_defaults()
    samplers = ["mh_sampler", "mtm_sampler", "smc_mh"]
    labels = ["MCMC-MH", "MTM", "SMC-MH"]

    function draw_samplers!(axis, samplers, labels)
        for (s, l) in zip(samplers, labels)
            df = collect_results(result_dir, rinclude=[Regex(s)])
            mean_seconds(trial) = mean(trial).time * 1e-9
            row = first(df)
            mean_sec = mean_seconds.(row.trials)
            # NOTE at ≈350 particles, the time per step triples. For 100x100 and 200x200 images. So it seems that CUDA or OpenGL struggles with textures of larger depth.
            MK.lines!(axis, row.n_particles, mean_sec; label=l)
        end
    end

    fig = MK.Figure(resolution=(DISS_WIDTH, 0.4 * DISS_WIDTH))
    ax = MK.Axis(fig[1, 1]; xlabel="number of particles", ylabel="mean step time / s")
    draw_samplers!(ax, samplers, labels)
    MK.axislegend(ax; position=:rb)

    ax2 = MK.Axis(fig; bbox=MK.BBox(80, 170, 105, 158), xticks=[0, 10, 20, 30], yticks=[0, 0.001, 0.002])
    draw_samplers!(ax2, samplers, labels)
    MK.limits!(ax2, 0, 30, 0, 0.002)

    # display(fig)
    save(joinpath("plots", "inference_time_$(img_size).pdf"), fig)
end
