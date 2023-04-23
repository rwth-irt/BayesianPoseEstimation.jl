# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using CUDA
using MCMCDepth
using Random
using Plots

CUDA.allowscalar(false)
gr()
MCMCDepth.diss_defaults()

parameters = Parameters()
# NOTE optimal parameter values of pixel_σ and normalization_constant seem to be inversely correlated. Moreover, different values seem to be optimal when using analytic association
@reset parameters.normalization_constant = 25
# NOTE Should be able to increase σ in MTM
@reset parameters.proposal_σ_r_quat = 0.5
@reset parameters.proposal_σ_t = [0.02, 0.02, 0.02]
# TODO same seed for experiments
@reset parameters.seed = rand(RandomDevice(), UInt32)
@reset parameters.n_steps = 800
@reset parameters.n_burn_in = 200
@reset parameters.n_thinning = 1
@reset parameters.n_particles = 100
# TODO tempering in MCMC?

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

# Model
prior = point_prior(gl_context, parameters, experiment, cpu_rng)
posterior = association_posterior(parameters, experiment, prior, dev_rng)
# posterior = simple_posterior(parameters, experiment, prior, dev_rng)
# posterior = smooth_posterior(parameters, experiment, prior, dev_rng)

# Sampler
# sampler = mh_sampler(cpu_rng, parameters, experiment, posterior)
# sampler = mh_local_sampler(cpu_rng, parameters, posterior)
sampler = mtm_sampler(cpu_rng, parameters, experiment, posterior)
# sampler = mtm_local_sampler(cpu_rng, parameters, posterior)

chain = sample(cpu_rng, posterior, sampler, parameters.n_steps; discard_initial=parameters.n_burn_in, thinning=parameters.n_thinning);
# TODO no conversion in smc? Also see next NOTE
# NOTE looks like sampling a pole which is probably sampling uniformly and transforming it back to Euler
plot_pose_chain(chain, 50)
plot_logprob(chain, 50)

# NOTE This does not look too bad. The most likely issue is the logjac correction which is calculated over all the pixels instead of the valid
plot_prob_img(mean_image(chain, :o))
plot_prob_img(chain[end].variables.o)

anim = @animate for i ∈ 0:2:360
    scatter_position(chain; camera=(i, 25), projection_type=:perspective, legend_position=:topright)
end;
gif(anim, "anim_fps15.gif", fps=20)
