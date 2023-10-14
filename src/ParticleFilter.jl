# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using KernelDistributions

function coordinate_pf_sampler(cpu_rng, params, posterior)
    # tempering does not matter for bootstrap kernel
    temp_schedule = ConstantSchedule()
    # TODO it is possible to use this interface but I really have to bend it to my will... redesign!
    t_proposal = Dynamics(:t, cpu_rng, params, posterior)
    r_proposal = Dynamics(:r, cpu_rng, params, posterior)
    t_kernel = BootstrapKernel(t_proposal)
    r_kernel = BootstrapKernel(r_proposal)
    CoordinateSampler(
        SequentialMonteCarlo(t_kernel, temp_schedule, params.n_particles, log(params.relative_ess * params.n_particles)),
        SequentialMonteCarlo(r_kernel, temp_schedule, params.n_particles, log(params.relative_ess * params.n_particles)))
end

function pf_sampler(cpu_rng, params, posterior)
    # tempering does not matter for bootstrap kernel
    temp_schedule = ConstantSchedule()
    # NOTE not component wise
    t_sym = BroadcastedNode(:t, cpu_rng, KernelNormal, 0, params.proposal_σ_t)
    r_sym = BroadcastedNode(:r, cpu_rng, KernelNormal, 0, params.proposal_σ_r)
    # TODO it is possible to use this interface but I really have to bend it to my will... redesign!
    tr_proposal = Proposal(propose_tr_dyn, transition_probability_symmetric, (; t=t_sym, r=r_sym), parents(posterior.prior, :r), (; t=ZeroIdentity(), r=ZeroIdentity()), bijector(posterior))
    tr_kernel = BootstrapKernel(tr_proposal)
    SequentialMonteCarlo(tr_kernel, temp_schedule, params.n_particles, log(params.relative_ess * params.n_particles))
end

"""
    Dynamics{name}
Proposal which uses state space dynamics for a given variable `name`.
This allows to include custom parameters like the velocity `decay`.
"""
struct Dynamics{name}
    rng
    decay
    σ
    bijectors
    evaluation
end

Dynamics(name::Symbol, rng::AbstractRNG, params::Parameters, posterior::PosteriorModel) = Dynamics{name}(rng, params.velocity_decay, params.proposal_σ_t, bijector(posterior), parents(posterior.prior, name))

transition_probability(dynamics::Dynamics, new_sample, previous_sample) = transition_probability_symmetric(dynamics, new_sample, previous_sample)

function propose(dynamics::Dynamics{:t}, sample, dims...)
    t = sample.variables.t
    t_d = sample.variables.t_dot
    t_dd = rand(dynamics.rng, KernelNormal.(0, dynamics.σ), dims...)
    # Decaying velocity
    @reset sample.variables.t_dot = dynamics.decay * t_d + t_dd
    # Constant acceleration integration
    @reset sample.variables.t = t + t_d + 0.5 * t_dd
    @reset sample.variables.t = t + t_dd
    # Evaluate rendering and possibly association
    model_sample, _ = to_model_domain(sample, dynamics.bijectors)
    evaluated = evaluate(dynamics.evaluation, variables(model_sample))
    to_unconstrained_domain(Sample(evaluated), dynamics.bijectors)
end

function propose(dynamics::Dynamics{:r}, sample, dims...)
    r = sample.variables.r
    r_d = sample.variables.r_dot
    r_dd = rand(dynamics.rng, KernelNormal.(0, dynamics.σ), dims...)
    # Decaying velocity
    @reset sample.variables.r_dot = dynamics.decay * r_d + r_dd
    # Constant acceleration integration
    @reset sample.variables.r = r .⊕ (r_d + 0.5 * r_dd)
    # Evaluate rendering
    model_sample, _ = to_model_domain(sample, dynamics.bijectors)
    evaluated = evaluate(dynamics.evaluation, variables(model_sample))
    to_unconstrained_domain(Sample(evaluated), dynamics.bijectors)
end

function propose_tr_dyn(proposal, sample, dims...)
    proposed = rand(proposal.model, dims...)

    t = sample.variables.t
    t_d = sample.variables.t_dot
    t_dd = proposed.t
    # TODO decay factor should be a parameter
    # Decaying velocity
    @reset sample.variables.t_dot = 0.8 * t_d + t_dd
    # Constant acceleration integration
    @reset sample.variables.t = t + t_d + 0.5 * t_dd
    @reset sample.variables.t = t + t_dd

    r = sample.variables.r
    r_d = sample.variables.r_dot
    r_dd = proposed.r
    # Decaying velocity
    @reset sample.variables.r_dot = 0.8 * r_d + r_dd
    # Constant acceleration integration
    @reset sample.variables.r = r .⊕ (r_d + 0.5 * r_dd)

    # Evaluate rendering
    model_sample, _ = to_model_domain(sample, proposal.posterior_bijectors)
    evaluated = evaluate(proposal.evaluation, variables(model_sample))
    to_unconstrained_domain(Sample(evaluated), proposal.posterior_bijectors)
end

function pf_inference(cpu_rng::AbstractRNG, dev_rng::AbstractRNG, posterior_fn, params::Parameters, experiment::Experiment, depth_imgs; collect_vars=(:t, :r))
    state = nothing
    states = Vector{SmcState}()
    for depth_img in depth_imgs
        # TODO crop depth_img
        experiment = Experiment(experiment, depth_img)
        prior = pose_prior(params, experiment, cpu_rng)
        posterior = posterior_fn(params, experiment, prior, dev_rng)
        # Bootstrap kernel for particle filter
        # sampler = smc_bootstrap(cpu_rng, params, posterior)
        # TODO allow different Samplers
        # NOTE component wise sampling is king, running twice allows much lower particle count
        sampler = coordinate_pf_sampler(cpu_rng, params, posterior)
        if isnothing(state)
            _, state = AbstractMCMC.step(cpu_rng, posterior, sampler)
        else
            _, state = AbstractMCMC.step(cpu_rng, posterior, sampler, state)
        end
        # TODO track ESS
        push!(states, collect_variables(state, collect_vars))
    end
    states, state
end