# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using DrWatson
@quickactivate("MCMCDepth")

using Accessors
using BenchmarkTools
using CUDA
using DataFrames
using MCMCDepth
using PoseErrors
using Random
using SciGL
# Avoid conflicts
import HyperTuning: HyperTuning, BCAPSampler, GridSampler, Scenario, history, best_parameters, (..), default_stop_criteria

using Logging
using ProgressLogging
using TerminalLoggers
global_logger(TerminalLogger(right_justify=120))

CUDA.allowscalar(false)
experiment_name = "mcmc_mh_hyperopt"
result_dir = datadir("exp_raw", experiment_name)
# Different hyperparameter for different datasets?
# NOTE ITODD might be influenced by different surface discrepancy threshold
dataset = ["lm", "itodd", "tless", "steri"]
testset = "train_pbr"
scene_id = 0
max_trials = 100
optsampler = :BCAPSampler
model = [:association_simple_reg, :smooth_posterior]
configs = dict_list(@dict dataset testset scene_id optsampler model max_trials)

"""
    rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row, config)
Assembles the posterior model and the sampler from the loaded images, mesh, and DataFrame row.
"""
function rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row, config)
    @unpack model = config
    # Context
    cpu_rng = Random.default_rng(parameters)
    dev_rng = device_rng(parameters)

    # Setup experiment
    camera = crop_camera(df_row)
    prior_o = fill(parameters.float_type(parameters.o_mask_not), parameters.width, parameters.height) |> device_array_type(parameters)
    prior_o[mask_img] .= parameters.o_mask_is
    # Prior t from mask is imprecise no need to bias
    prior_t = point_from_segmentation(df_row.bbox, depth_img, mask_img, df_row.cv_camera)
    experiment = Experiment(gl_context, Scene(camera, [mesh]), prior_o, prior_t, depth_img)

    # Model
    prior = point_prior(parameters, experiment, cpu_rng)
    posterior = eval(model)(parameters, experiment, prior, dev_rng)
    # Sampler
    sampler = mh_sampler(cpu_rng, parameters, posterior)
    # Result
    cpu_rng, posterior, sampler
end

"""
    timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row, config)
Report the wall time from the point right after the raw data (the image, 3D object models etc.) is loaded.
"""
function timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row, config)
    time = @elapsed begin
        # Assemble sampler and run inference
        rng, posterior, sampler = rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row, config)
        # Only MTM supports multiple particles. Metropolis Hastings in recall_n_steps.jl
        chain = sample(rng, posterior, sampler, parameters.n_steps; discard_initial=parameters.n_burn_in, thinning=parameters.n_thinning, progress=false)
        # Extract best pose and score
        score, idx = findmax(loglikelihood.(chain))
        t = variables(chain[idx]).t
        r = variables(chain[idx]).r
    end
    t, r, score, time
end

"""
    cost_function(parameters, gl_context, config, scene_df)
Returns (1 - VSD recall) as the costs for the hyperparameter optimization.
Pass the scene_df to avoid loading it twice.
"""
function cost_function(parameters, gl_context, config, scene_df)
    HyperTuning.@unpack dataset, testset, scene_id = config
    # Store result in DataFrame. Numerical precision doesn't matter here → Float32
    est_df = select(scene_df, :scene_id, :img_id, :obj_id)
    est_df.score = Vector{Float32}(undef, nrow(est_df))
    est_df.R = Vector{Quaternion{Float32}}(undef, nrow(est_df))
    est_df.t = Vector{Vector{Float32}}(undef, nrow(est_df))
    est_df.time = Vector{Float32}(undef, nrow(est_df))

    # Run inference per detection
    @progress "dataset: $dataset, scene_id: $scene_id" for (idx, df_row) in enumerate(eachrow(scene_df))
        # Image crops differ per object
        depth_img, mask_img, mesh = load_img_mesh(df_row, parameters, gl_context)
        # Run and collect results
        t, R, score, time = timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row, config)
        est_df[idx, :].score = score
        est_df[idx, :].R = R
        est_df[idx, :].t = t
        est_df[idx, :].time = time
    end

    # Add gt_R & gt_t for testset
    gt_df = bop_test_or_train(dataset, testset, scene_id)
    datasubset_path = datadir("bop", dataset, testset)
    if !("gt_t" in names(gt_df))
        leftjoin!(gt_df, PoseErrors.gt_dataframe(datasubset_path, scene_id)
            ; on=[:scene_id, :img_id, :gt_id])
    end
    if !("visib_fract" in names(gt_df))
        leftjoin!(gt_df, PoseErrors.gt_info_dataframe(datasubset_path, scene_id); on=[:scene_id, :img_id, :gt_id])
    end
    df = outerjoin(gt_df, est_df; on=[:scene_id, :img_id, :obj_id])

    # Keep only visibility fraction >= 0.1
    filter!(:visib_fract => (x -> x >= 0.1), df)
    # Only estimates for which a ground truth exists are relevant for the recall
    filter!(:gt_t => (x -> !ismissing(x)), df)

    # Different VSD δ for visible surface in ITODD & Steri
    vsd_δ = contains(dataset, "itodd") || contains(dataset, "steri") ? ITODD_δ : BOP_δ |> Float32
    # WARN do not parallelize using ThreadsX, OpenGL is sequential
    # WARN this should be a distance context but two contexts at the same time do not work ... error must only be proportional for hyperparameter optimization
    df.vsd = map(row -> vsd_depth_row(row, gl_context, vsd_δ), eachrow(df))

    # Greedy matching of the ground truth to the estimates
    errors_per_obj = groupby(df, [:scene_id, :img_id, :obj_id])
    matched_df = combine(match_obj_errors, errors_per_obj; threads=false)
    # must be the number of ground truth annotations
    @assert nrow(matched_df) == nrow(unique(df, :gt_t))

    # Calc recall
    vsd_thresh = threshold_errors.(matched_df.vsd, BOP18_θ)
    1 - recall(vsd_thresh...)
end

"""
run_hyperopt(config)
    Save results per case via DrWatson's produce_or_load for the `config`
"""
function run_hyperopt(config)
    # Extract config and load dataset
    @unpack dataset, testset, scene_id, model, optsampler, max_trials = config
    scene_df = bop_test_or_train(dataset, testset, scene_id)
    parameters = Parameters()
    # Finally destroy OpenGL context
    gl_context = render_context(parameters)
    try
        # Benchmark model sampler configuration to adapt number of steps - also avoids timing pre-compilation
        df_row = first(scene_df)
        depth_img, mask_img, mesh = load_img_mesh(df_row, parameters, gl_context)
        rng, posterior, sampler = rng_posterior_sampler(gl_context, parameters, depth_img, mask_img, mesh, df_row, config)
        step_time = mean_step_time(rng, posterior, sampler)
        @reset parameters.n_steps = floor(Int, parameters.time_budget / step_time)

        # Capture local parameters in this closure which suffices the HyperTuning interface
        function objective(trial)
            @unpack c_reg, proposal_σ_r, proposal_σ_t, pixel_σ = trial
            @reset parameters.c_reg = c_reg
            # NOTE does not make sense to optimize heavily correlated variables e.g. σ_t & c_reg
            @reset parameters.proposal_σ_r = fill(proposal_σ_r, 3)
            @reset parameters.proposal_σ_t = fill(proposal_σ_t, 3)
            @reset parameters.pixel_σ = pixel_σ
            @reset parameters.association_σ = pixel_σ
            cost_function(parameters, gl_context, config, scene_df)
        end
        scenario = Scenario(
            c_reg=(5.0 .. 100.0),
            pixel_σ=(0.001 .. 0.1),
            proposal_σ_r=(0.01 .. 1.0),
            proposal_σ_t=(0.001 .. 0.1),
            sampler=eval(optsampler)(),
            max_trials=max_trials,
            batch_size=1    # No support for multiple OpenGL contexts
        )
        @progress "Optimizer: $optsampler Model: $model " for _ in 1:max_trials
            if default_stop_criteria(scenario)
                break
            end
            HyperTuning.optimize!(objective, scenario)
        end
        Dict("scenario" => scenario)
    finally
        # If not destroyed, weird stuff happens
        destroy_context(gl_context)
    end
end

@progress "Hyperparameter optimization" for config in configs
    @produce_or_load(run_hyperopt, config, result_dir; filename=my_savename)
end

# TODO analyze results on a per-dataset basis. Different scenes - different parameters?
pro_df = collect_results(result_dir)

for row in eachrow(pro_df)
    scenario = row.scenario
    hist = history(scenario)
    println(row.path)
    show(best_parameters(scenario))
end


# TODO analyze and plot results on validation set, scene 1-4
