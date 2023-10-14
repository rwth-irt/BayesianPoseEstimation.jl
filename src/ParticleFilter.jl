# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using KernelDistributions

function pf_sampler(cpu_rng, params, posterior)
    # tempering does not matter for bootstrap kernel
    temp_schedule = ConstantSchedule()
    # NOTE not component wise
    t_sym = BroadcastedNode(:t, cpu_rng, KernelNormal, 0, params.proposal_σ_t)
    r_sym = BroadcastedNode(:r, cpu_rng, KernelNormal, 0, params.proposal_σ_r)
    # TODO it is possible to use this interface but I really have to bend it to my will... redesign!
    t_proposal = Proposal(propose_t_dyn, transition_probability_symmetric, (; t=t_sym), parents(posterior.prior, :t), (; t=ZeroIdentity()), bijector(posterior))
    r_proposal = Proposal(propose_r_dyn, transition_probability_symmetric, (; r=r_sym), parents(posterior.prior, :r), (; r=ZeroIdentity()), bijector(posterior))
    t_kernel = BootstrapKernel(t_proposal)
    r_kernel = BootstrapKernel(r_proposal)
    CoordinateSampler(
        SequentialMonteCarlo(t_kernel, temp_schedule, params.n_particles, log(params.relative_ess * params.n_particles)),
        SequentialMonteCarlo(r_kernel, temp_schedule, params.n_particles, log(params.relative_ess * params.n_particles)))
end

function propose_t_dyn(proposal, sample, dims...)
    t = sample.variables.t
    t_d = sample.variables.t_dot
    t_dd = rand(proposal.model, dims...).t
    # Decaying velocity
    @reset sample.variables.t_dot = 0.9 * (t_d + t_dd)
    # Constant acceleration integration
    @reset sample.variables.t = t + t_d + 0.5 * t_dd
    @reset sample.variables.t = t + t_dd
    # Evaluate rendering
    model_sample, _ = to_model_domain(sample, proposal.posterior_bijectors)
    evaluated = evaluate(proposal.evaluation, variables(model_sample))
    to_unconstrained_domain(Sample(evaluated), proposal.posterior_bijectors)

end

function propose_r_dyn(proposal, sample, dims...)
    r = sample.variables.r
    r_d = sample.variables.r_dot
    r_dd = rand(proposal.model, dims...).r
    # Decaying velocity
    @reset sample.variables.r_dot = 0.9 * r_d + r_dd
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
        # TODO prior for orientation, too;
        prior = pose_prior(params, experiment, cpu_rng)
        # TODO or association / smooth
        posterior = posterior_fn(params, experiment, prior, dev_rng)
        # Bootstrap kernel for particle filter
        # sampler = smc_bootstrap(cpu_rng, params, posterior)
        sampler = pf_sampler(cpu_rng, params, posterior)
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