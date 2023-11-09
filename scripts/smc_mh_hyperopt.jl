# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using DrWatson
@quickactivate("MCMCDepth")

using Accessors
using BenchmarkTools
using CUDA
using CSV
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
experiment_name = "smc_mh_hyperopt"
result_dir = datadir("exp_raw", experiment_name)
# Different hyperparameter for different datasets?
# NOTE ITODD might be influenced by different surface discrepancy threshold
dataset = ["lm", "itodd", "tless", "steri"]
testset = "train_pbr"
scene_id = 0
max_evals = 200
optsampler = :BCAPSampler
# smooth posterior won't perform any better
model = :simple_posterior # [:simple_posterior, :smooth_posterior]
configs = dict_list(@dict dataset testset scene_id optsampler model max_evals)

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
    prior_o[mask_img] .= parameters.float_type(parameters.o_mask_is)
    # Prior t from mask is imprecise no need to bias
    prior_t = point_from_segmentation(df_row.bbox, depth_img, mask_img, df_row.cv_camera)
    experiment = Experiment(gl_context, Scene(camera, [mesh]), prior_o, prior_t, depth_img)

    # Model
    prior = point_prior(parameters, experiment, cpu_rng)
    posterior = eval(model)(parameters, experiment, prior, dev_rng)
    # Sampler
    sampler = smc_mh(cpu_rng, parameters, posterior)
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
        states, final_state = smc_inference(rng, posterior, sampler, parameters)

        # Extract best pose and score
        final_sample = final_state.sample
        score, idx = findmax(loglikelihood(final_sample))
        t = variables(final_sample).t[:, idx]
        r = variables(final_sample).r[idx]
    end
    t, r, score, final_state, states, time
end

"""
    cost_function(parameters, gl_context, config, scene_df)
Returns (1 - VSD recall) as the costs for the hyperparameter optimization.
Pass the scene_df to avoid loading it twice.
"""
function cost_function(parameters, gl_context, config, scene_df)
    @unpack dataset, testset, scene_id = config
    # Store result in DataFrame. Numerical precision doesn't matter here → Float32
    est_df = select(scene_df, :scene_id, :img_id, :obj_id)
    est_df.score = Vector{Float32}(undef, nrow(est_df))
    est_df.R = Vector{Quaternion{Float32}}(undef, nrow(est_df))
    est_df.t = Vector{Vector{Float32}}(undef, nrow(est_df))
    est_df.time = Vector{Float32}(undef, nrow(est_df))
    est_df.final_state = Vector{SmcState}(undef, nrow(est_df))
    est_df.log_evidence = Vector{Vector{Float32}}(undef, nrow(est_df))

    # Run inference per detection
    @progress "dataset: $dataset, scene_id: $scene_id" for (idx, df_row) in enumerate(eachrow(scene_df))
        # Image crops differ per object
        depth_img, mask_img, mesh = load_img_mesh(df_row, parameters, gl_context)
        # Run and collect results
        t, R, score, final_state, states, time = timed_inference(gl_context, parameters, depth_img, mask_img, mesh, df_row, config)
        # Avoid too large files by only saving t, r, and the logevidence not the sequence of states
        final_state = collect_variables(final_state, (:t, :r))
        # Avoid out of GPU errors
        @reset final_state.sample = to_cpu(final_state.sample)
        est_df[idx, :].score = score
        est_df[idx, :].R = R
        est_df[idx, :].t = t
        est_df[idx, :].time = time
        est_df[idx, :].final_state = final_state
        est_df[idx, :].log_evidence = logevidence.(states)
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
    @unpack dataset, testset, scene_id, model, optsampler, max_evals = config
    scene_df = bop_test_or_train(dataset, testset, scene_id)
    parameters = Parameters()
    if dataset == "steri"
        @reset parameters.width = 60
        @reset parameters.height = 60
    end
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
            @unpack o_mask_is, pixel_σ, proposal_σ_r = trial
            @reset parameters.o_mask_is = o_mask_is
            @reset parameters.o_mask_not = 1 - o_mask_is
            # NOTE does not make sense to optimize heavily correlated variables e.g. σ_t & c_reg
            # @reset parameters.c_reg = c_reg
            @reset parameters.proposal_σ_r = fill(proposal_σ_r, 3)
            @reset parameters.pixel_σ = pixel_σ
            @reset parameters.association_σ = pixel_σ
            cost_function(parameters, gl_context, config, scene_df)
        end
        scenario = Scenario(
            o_mask_is=(0.5f0 .. 1.0f0),
            pixel_σ=(0.0001f0 .. 0.02f0),
            proposal_σ_r=(0.01f0 .. Float32(π)),
            # Not really interesting?
            # c_reg=(1 .. 500),
            sampler=eval(optsampler)(),
            max_evals=max_evals,
            max_trials=max_evals,
            batch_size=1,    # No support for multiple OpenGL contexts
            verbose=true
        )
        @progress "Optimizer: $optsampler Model: $model " for _ in 1:max_evals
            if default_stop_criteria(scenario)
                break
            end
            HyperTuning.optimize!(objective, scenario)
        end
        Dict("scenario" => scenario, "parameters" => parameters)
    finally
        # If not destroyed, weird stuff happens
        destroy_context(gl_context)
    end
end

@progress "Hyperparameter optimization" for config in configs
    @produce_or_load(run_hyperopt, config, result_dir; filename=my_savename)
end

function parse_config(path)
    config = my_parse_savename(path)
    @unpack dataset, model = config
    dataset, model
end
pro_df = collect_results(result_dir)
transform!(pro_df, :path => ByRow(parse_config) => [:dataset, :model])
pro_dir = datadir("exp_pro", experiment_name)
mkpath(pro_dir)
groups = groupby(pro_df, :model)
for (key, group) in zip(keys(groups), groups)
    res_df = DataFrame(dataset=String[], o_mask_is=Float64[], pixel_σ=Float64[], proposal_σ_r=Float64[], vsd_recall=[])
    for row in eachrow(group)
        best = best_parameters(row.scenario)
        println(row.dataset)
        display(best)
        recall = 1 - best.performance
        vals = best.values
        push!(res_df, (; dataset=row.dataset, o_mask_is=vals[:o_mask_is], pixel_σ=vals[:pixel_σ], proposal_σ_r=vals[:proposal_σ_r], vsd_recall=recall))
    end
    display(res_df)
    CSV.write(joinpath(pro_dir, "$(key.model).csv"), res_df)
end
# TODO exclude steri when calculating mean
