# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

using AbstractMCMC: step
using Accessors
using CUDA
using MCMCDepth
using Random
using Plots
using ProgressLogging

CUDA.allowscalar(false)
gr()
MCMCDepth.diss_defaults()

parameters = Parameters()
# NOTE SMC: tempering is essential. More steps (MCMC) allows higher normalization_constant than more particles (FP, Bootstrap), 15-30 seems to be a good range
@reset parameters.normalization_constant = 25;
@reset parameters.proposal_σ_r_quat = 0.1;
@reset parameters.proposal_σ_t = [0.01, 0.01, 0.01];
# TODO same seed for experiments
@reset parameters.seed = rand(RandomDevice(), UInt32);
# NOTE resampling dominated like FP & Bootstrap kernels typically perform better with more samples (1_000,100) while MCMC kernels tend to perform better with more steps (2_000,50)
@reset parameters.n_steps = 1_000
@reset parameters.n_particles = 50
# Normalization and tempering leads to less resampling, especially in MCMC sampler
@reset parameters.relative_ess = 0.8;

# NOTE takes minutes instead of seconds
# @reset parameters.device = :CPU
cpu_rng = rng(parameters)
dev_rng = device_rng(parameters)
gl_context = render_context(parameters)

include("fake_observation.jl")
camera = fake_camera(parameters)
model = upload_mesh(gl_context, fake_mesh)
prior_t = fake_gt_position + [0.05, -0.05, -0.1]
fake_img = fake_observation(gl_context, parameters, 0.4)
experiment = Experiment(Scene(camera, [model]), prior_t, fake_img)

# TODO implement for AbstractSampler? specialize sample? specialize step? Both solutions are pretty inconvenient since the type cannot easily be inferred for ComposedSampler (SMC vs. MCMC) → name functions
function smc_inference(rng, posterior, sampler, params::Parameters)
    sample, state = step(rng, posterior, sampler)
    @progress for _ in 1:params.n_steps
        sample, state = step(rng, posterior, sampler, state)
    end
    sample, state
end

# Model
prior = point_prior(gl_context, parameters, experiment, cpu_rng)
# posterior = association_posterior(parameters, experiment, prior, dev_rng)
posterior = simple_posterior(parameters, experiment, prior, dev_rng)
# posterior = smooth_posterior(parameters, experiment, prior, dev_rng)

# Sampler
sampler = smc_mh(cpu_rng, parameters, experiment, posterior)
# sampler = smc_bootstrap(cpu_rng, parameters, posterior)
# sampler = smc_forward(cpu_rng, parameters, posterior)

# NOTE Benchmark results for smc_mh, 1_000 steps & 50 particles
# association_posterior ~ 1.51sec, simple_posterior ~ 1.15sec & smooth_posterior ~ 1.53sec (all±40mss)
# smc_forward simple_posterior ~ 1.08sec \pm
final_sample, final_state = smc_inference(cpu_rng, posterior, sampler, parameters);
println("Final log-evidence: $(final_state.log_evidence)")
plot_pose_density(final_sample; trim=false)

anim = @animate for i ∈ 0:2:360
    scatter_position(final_sample, 100, label="particle number", camera=(i, 25), projection_type=:perspective, legend_position=:topright)
end;
gif(anim, "anim_fps15.gif", fps=20)
# TODO diagnostics: Accepted steps, resampling steps
