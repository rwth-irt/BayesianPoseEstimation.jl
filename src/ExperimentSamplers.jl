# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

import Distributions

# hack Distributions.jl to allow logdensityof for MvNormal and multiple samples
DensityInterface.logdensityof(d::Distributions.MvNormal, x::AbstractMatrix) = Distributions.logpdf(d, x)

"""
    mh_sampler(cpu_rng, params, experiment, posterior)
Component-wise sampling of the position and orientation via Metropolis-Hastings.
With a low probability (~1%) the sample is drawn independently from the prior a to avoid local minima.
"""
function mh_sampler(cpu_rng, params, experiment, posterior)
    t_ind = BroadcastedNode(:t, cpu_rng, KernelNormal, experiment.prior_t, params.σ_t)
    r_ind = BroadcastedNode(:r, cpu_rng, QuaternionUniform, params.float_type)
    t_ind_proposal = independent_proposal(t_ind, posterior)
    r_ind_proposal = independent_proposal(r_ind, posterior)

    t_sym = BroadcastedNode(:t, cpu_rng, KernelNormal, 0, params.proposal_σ_t)
    r_sym = BroadcastedNode(:r, cpu_rng, KernelNormal, 0, params.proposal_σ_r)
    t_sym_proposal = symmetric_proposal(t_sym, posterior)
    r_sym_proposal = symmetric_proposal(r_sym, posterior)

    proposals = (t_sym_proposal, r_sym_proposal, t_ind_proposal, r_ind_proposal)
    weights = Weights([1.0, 1.0, 0.01, 0.01])
    samplers = map(proposals) do proposal
        MetropolisHastings(proposal)
    end
    ComposedSampler(weights, samplers...)
end

"""
    mh_sampler(cpu_rng, params, posterior)
Component-wise sampling of the position and orientation via Metropolis-Hastings.
Local moves only, no sample is drawn independently from the prior.
"""
function mh_local_sampler(cpu_rng, params, posterior)
    t_sym = BroadcastedNode(:t, cpu_rng, KernelNormal, 0, params.proposal_σ_t)
    r_sym = BroadcastedNode(:r, cpu_rng, KernelNormal, 0, params.proposal_σ_r)
    t_sym_proposal = symmetric_proposal(t_sym, posterior)
    r_sym_proposal = symmetric_proposal(r_sym, posterior)

    proposals = (t_sym_proposal, r_sym_proposal)
    weights = Weights([1.0, 1.0])
    samplers = map(proposals) do proposal
        MetropolisHastings(proposal)
    end
    ComposedSampler(weights, samplers...)
end

"""
    mh_sampler(cpu_rng, params, experiment, posterior)
Component-wise sampling of the position and orientation via Multiple-Try-Metropolis.
With a low probability (~1%) the sample is drawn independently from the prior a to avoid local minima.
"""
function mtm_sampler(cpu_rng, params, experiment, posterior)
    t_ind = BroadcastedNode(:t, cpu_rng, KernelNormal, experiment.prior_t, params.σ_t)
    r_ind = BroadcastedNode(:r, cpu_rng, QuaternionUniform, params.float_type)
    t_ind_proposal = independent_proposal(t_ind, posterior)
    r_ind_proposal = independent_proposal(r_ind, posterior)

    t_sym = BroadcastedNode(:t, cpu_rng, KernelNormal, 0, params.proposal_σ_t)
    r_sym = BroadcastedNode(:r, cpu_rng, KernelNormal, 0, params.proposal_σ_r)
    t_sym_proposal = symmetric_proposal(t_sym, posterior)
    r_sym_proposal = symmetric_proposal(r_sym, posterior)

    proposals = (t_sym_proposal, r_sym_proposal, t_ind_proposal, r_ind_proposal)
    weights = Weights([1.0, 1.0, 0.01, 0.01])
    samplers = map(proposals) do proposal
        MultipleTry(proposal, params.n_particles)
    end
    ComposedSampler(weights, samplers...)
end

"""
    mh_sampler(cpu_rng, params, posterior)
Component-wise sampling of the position and orientation via Multiple-Try-Metropolis.
Local moves only, no sample is drawn independently from the prior.
"""
function mtm_local_sampler(cpu_rng, params, posterior)
    t_sym = BroadcastedNode(:t, cpu_rng, KernelNormal, 0, params.proposal_σ_t)
    r_sym = BroadcastedNode(:r, cpu_rng, KernelNormal, 0, params.proposal_σ_r)
    t_sym_proposal = symmetric_proposal(t_sym, posterior)
    r_sym_proposal = symmetric_proposal(r_sym, posterior)

    proposals = (t_sym_proposal, r_sym_proposal)
    weights = Weights([1.0, 1.0])
    samplers = map(proposals) do proposal
        MultipleTry(proposal, params.n_particles)
    end
    ComposedSampler(weights, samplers...)
end

"""
    smc_forward(cpu_rng, params, posterior)
Component-wise sampling of the position and orientation via Sequential Monte Carlo with a forward proposal kernel which results in weights similar to the MH acceptance ratio.
Local moves only, no sample is drawn independently from the prior.
"""
function smc_forward(cpu_rng, params, posterior)
    temp_schedule = LinearSchedule(params.n_steps)
    t_sym = BroadcastedNode(:t, cpu_rng, KernelNormal, 0, params.proposal_σ_t)
    r_sym = BroadcastedNode(:r, cpu_rng, KernelNormal, 0, params.proposal_σ_r)
    t_sym_proposal = symmetric_proposal(t_sym, posterior)
    r_sym_proposal = symmetric_proposal(r_sym, posterior)
    t_sym_kernel = AdaptiveKernel(cpu_rng, ForwardProposalKernel(t_sym_proposal))
    r_sym_kernel = ForwardProposalKernel(r_sym_proposal)

    kernels = (t_sym_kernel, r_sym_kernel)
    weights = Weights([1.0, 1.0])
    samplers = map(kernels) do kernel
        SequentialMonteCarlo(kernel, temp_schedule, params.n_particles, log(params.relative_ess * params.n_particles))
    end
    ComposedSampler(weights, samplers...)
end

# NOTE tends to diverge with to few samples, since there is no prior pulling it back to sensible values. But it can also converge with over-confident variance since there is no prior holding it back.
"""
    smc_bootstrap(cpu_rng, params, posterior)
Component-wise sampling of the position and orientation via Sequential Monte Carlo with a bootstrap kernel which results in the loglikelihood as weights similar to a bootstrap particle filter.
Local moves only, no sample is drawn independently from the prior.
"""
function smc_bootstrap(cpu_rng, params, posterior)
    temp_schedule = LinearSchedule(params.n_steps)
    t_sym = BroadcastedNode(:t, cpu_rng, KernelNormal, 0, params.proposal_σ_t)
    r_sym = BroadcastedNode(:r, cpu_rng, KernelNormal, 0, params.proposal_σ_r)
    t_sym_proposal = symmetric_proposal(t_sym, posterior)
    r_sym_proposal = symmetric_proposal(r_sym, posterior)
    t_sym_kernel = AdaptiveKernel(cpu_rng, ForwardProposalKernel(t_sym_proposal))
    r_sym_kernel = ForwardProposalKernel(r_sym_proposal)

    kernels = (t_sym_kernel, r_sym_kernel)
    weights = Weights([1.0, 1.0])
    samplers = map(kernels) do kernel
        SequentialMonteCarlo(kernel, temp_schedule, params.n_particles, log(params.relative_ess * params.n_particles))
    end
    ComposedSampler(weights, samplers...)
end

"""
    smc_mh(cpu_rng, params, posterior)
Component-wise sampling of the position and orientation via Sequential Monte Carlo with a Metropolis Hastings kernel which uses a likelihood-tempered weight update.
Thanks to the Metropolis Hastings kernel which only replaces a subset of the samples, samples are drawn with a low probability (~5%) from the prior.
Use this sample for exploration.
"""
function smc_mh(cpu_rng, params, posterior)
    # NOTE LinearSchedule seems reasonable, ExponentialSchedule and ConstantSchedule either explore too much or not enough
    temp_schedule = LinearSchedule(params.n_steps)

    # NOTE use independent proposals only with an MCMC Kernel, otherwise all information is thrown away.
    r_ind = BroadcastedNode(:r, cpu_rng, QuaternionUniform, params.float_type)
    r_ind_proposal = independent_proposal(r_ind, posterior)
    r_ind_kernel = MhKernel(cpu_rng, r_ind_proposal)

    t_sym = BroadcastedNode(:t, cpu_rng, KernelNormal, 0, params.proposal_σ_t)
    r_sym = BroadcastedNode(:r, cpu_rng, KernelNormal, 0, params.proposal_σ_r)
    t_sym_proposal = symmetric_proposal(t_sym, posterior)
    r_sym_proposal = symmetric_proposal(r_sym, posterior)
    # NOTE adaptive rotations proposals do not work well since the rotation are usually not normally distributed.
    t_sym_kernel = AdaptiveKernel(cpu_rng, MhKernel(cpu_rng, t_sym_proposal))
    r_sym_kernel = MhKernel(cpu_rng, r_sym_proposal)

    # TODO o needs dev_rng
    # o_sym = BroadcastedNode(:o, CUDA.default_rng(), KernelNormal, 0.0f0, 0.1f0)
    # o_sym_proposal = symmetric_proposal(o_sym, posterior)

    # NOTE t_ind should not be required since it is quite local and driven via the adaptive variance
    kernels = (t_sym_kernel, r_sym_kernel, r_ind_kernel)
    weights = Weights([1.0, 1.0, 0.1])
    samplers = map(kernels) do kernel
        SequentialMonteCarlo(kernel, temp_schedule, params.n_particles, log(params.relative_ess * params.n_particles))
    end
    ComposedSampler(weights, samplers...)
end

# TODO loop until time budget is up? Not a good idea, might be disturbed by updates etc. Before each experiment run a short benchmark and calculate steps/sec and use this to calculate n_steps for the experiment
"""
    smc_inference(cpu_rng, posterior, sampler, params; [collect_vars=(:t, :r)])
Run the inference iterations and return `(states, final_state)`.
Use `collect_vars` to specify which variables to collect in `states`, e.g. to avoid out of GPU memory errors.
"""
function smc_inference(cpu_rng, posterior, sampler, params::Parameters; collect_vars=(:t, :r))
    states = Vector{SmcState}(undef, params.n_steps)
    _, state = AbstractMCMC.step(cpu_rng, posterior, sampler)
    states[1] = collect_variables(state, collect_vars)
    @progress for idx in 2:params.n_steps
        _, state = AbstractMCMC.step(cpu_rng, posterior, sampler, state)
        collect_state = @set state.sample.variables = state.sample.variables[collect_vars]
        states[idx] = collect_variables(state, collect_vars)
    end
    states, state
end

collect_variables(state::SmcState, var_names) = @set state.sample.variables = state.sample.variables[var_names]
