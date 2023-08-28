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

@info "Loading packages"
using Accessors
using BenchmarkTools
using CUDA
using DataFrames
using MCMCDepth
using Plots
using Plots.PlotMeasures
using Random
using SciGL

using Logging
using ProgressLogging
using TerminalLoggers
global_logger(TerminalLogger(right_justify=120))

# Experiment setup
experiment_name = "inference_time"
sampler = [:mtm_sampler, :smc_bootstrap, :smc_forward, :smc_mh]
configs = dict_list(@dict sampler)

# Context
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1
CUDA.allowscalar(false)
pythonplot()
diss_defaults()
parameters = Parameters()
cpu_rng = Random.default_rng(parameters)
dev_rng = device_rng(parameters)

@reset parameters.width = 200
@reset parameters.height = 200
@reset parameters.depth = 500
@reset parameters.device = :CUDA
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
mask_img = depth_img .> 0

# Probabilistic model
experiment = Experiment(gl_context, Scene(camera, [mesh]), mask_img, pose.translation.translation, depth_img)
prior = point_prior(parameters, experiment, cpu_rng)
posterior = simple_posterior(parameters, experiment, prior, dev_rng)

function run_experiment(posterior, parameters, config)
    cpu_rng = Random.default_rng(parameters)
    @unpack sampler = config

    n_particles = [1, 10:10:500...]
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
@progress "inference time" for config in configs
    @produce_or_load(run_closure, config, result_dir; filename=c -> savename(c; connector=","))
end
destroy_context(gl_context)

# Visualize and save plot
samplers = ["mtm_sampler", "smc_bootstrap", "smc_forward", "smc_mh"]
labels = ["MTM", "SMC bootstrap", "SMC forward", "SMC MH"]

p = plot(; legend=:topleft)
for (s, l) in zip(samplers, labels)
    df = collect_results(result_dir, rinclude=[Regex(s)])
    mean_seconds(trial) = mean(trial).time * 1e-9
    row = first(df)
    # NOTE at ≈350 particles, the time per step triples. For 100x100 and 200x200 images. So it seems that CUDA or OpenGL struggles with textures of larger depth.
    plot!(row.n_particles, mean_seconds.(row.trials); xlabel="number of particles", ylabel="mean step time / s", label=l)
end

plot!(; yticks=false, legend=false, inset=(1, bbox(2mm, 0.4, 0.35, 0.35, :left)), subplot=2)
for (s, l) in zip(samplers, labels)
    # s, l = first(samplers), first(labels)
    df = collect_results(result_dir, rinclude=[Regex(s)])
    mean_seconds(trial) = mean(trial).time * 1e-9
    row = first(df)
    # NOTE at ≈350 particles, the time per step triples. For 100x100 and 200x200 images. So it seems that CUDA or OpenGL struggles with textures of larger depth.
    # plot!(p, row.n_particles[1:5], mean_seconds.(row.trials)[1:5]; legend=false, tickfontcolor=:transparent, inset=(1, bbox(0, 0, 0.3, 0.3, :bottom, :right)), subplot=1)
    plot!(p[2], row.n_particles[1:4], mean_seconds.(row.trials)[1:4])
end
display(p)
savefig(p, joinpath("plots", "inference_time.svg"))
