# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using BayesNet
using KernelDistributions
using MCMCDepth
using Random
using Test

rng = Random.default_rng()

a = BroadcastedNode(:a, rng, KernelNormal, 1.0f0, 2.0f0)
b = BroadcastedNode(:b, rng, KernelExponential, [1.0f0, 3.0f0])
c = BroadcastedNode(:c, rng, KernelNormal, (a, b))
data = rand(c, 1, 50).c
c_obs = c | data
posterior_model = PosteriorModel(c_obs)

sample = @inferred rand(posterior_model)
ℓ = @inferred logdensityof(posterior_model, sample)
@test ℓ isa Float32
sample = @inferred rand(posterior_model, 5)
ℓ = @inferred logdensityof(posterior_model, sample)
@test ℓ isa AbstractArray{Float32,1}
@test size(ℓ) == (5,)

@testset "Type stable ModifierNode" begin
  # Had issues with model(::ModifierNode) being unstable when MCMCDepth was loaded.
  sample = @inferred rand(posterior_model)
  ℓ = @inferred logdensityof(posterior_model, sample)
  @test ℓ isa Float32
end

proposal_model = (; a=BroadcastedNode(:a, rng, KernelNormal, Float32), b=BroadcastedNode(:b, rng, KernelNormal, [0.0f0, 0.0f0], [1.0f0, 1.0f0]))
proposal = symmetric_proposal(proposal_model, posterior_model)

kernels = (ForwardProposalKernel(proposal), BootstrapKernel(proposal), MhKernel(rng, proposal), AdaptiveKernel(rng, ForwardProposalKernel(proposal)), AdaptiveKernel(rng, BootstrapKernel(proposal)), AdaptiveKernel(rng, MhKernel(rng, proposal)))

@testset "SMC kernel: $(kernel |> typeof |> nameof)" for kernel in kernels
  n_steps = 42
  n_particles = 6
  smc = SequentialMonteCarlo(kernel, LinearSchedule(n_steps), n_particles, log(0.5 * n_particles))

  sample, state = @inferred AbstractMCMC.step(rng, posterior_model, smc)

  @test sample.variables.a isa AbstractArray{Float32}
  @test sample.variables.b isa AbstractArray{Float32}
  # logp will change to Float64 due to Tempering

  @test size(sample.variables.a) == (6,)
  @test size(sample.variables.b) == (2, 6)
  @test size(sample.logp) == (6,)

  @test state.log_evidence == 0
  @test size(state.log_likelihood) == (6,)
  @test size(state.log_weights) == (6,)
  @test state.sample == sample
  @test state.temperature == 0

  sample, state = AbstractMCMC.step(rng, posterior_model, smc, state)

  @test sample.variables.a isa AbstractArray{Float32}
  @test sample.variables.b isa AbstractArray{Float32}

  @test size(sample.variables.a) == (6,)
  @test size(sample.variables.b) == (2, 6)
  @test size(sample.logp) == (6,)

  @test state.log_evidence != 0
  @test state.log_likelihood isa AbstractArray{Float32}
  @test state.log_weights isa AbstractArray{Float64}
  @test state.sample == sample
  @test state.temperature > 0
end;

n_steps = 42
n_particles = 6
smc = SequentialMonteCarlo(kernel, LinearSchedule(n_steps), n_particles, log(0.5 * n_particles))

@testset "adaptive_mvnormal" begin
  _, state = @inferred AbstractMCMC.step(rng, posterior_model, smc)

  # Univariate - KernelNormal
  # TODO Do I want to try to make it type stable?
  proposal = symmetric_proposal(BroadcastedNode(:a, rng, KernelNormal, Float32), posterior_model)
  res = MCMCDepth.adaptive_mvnormal(rng, proposal, state)
  @test res.model != proposal.model
  @test res.model.a.model isa KernelNormal

  # Multivariate - MvNormal
  proposal = symmetric_proposal(BroadcastedNode(:b, rng, KernelNormal, [0.0f0, 0.0f0], [1.0f0, 1.0f0]), posterior_model)
  res = MCMCDepth.adaptive_mvnormal(rng, proposal, state)
  @test res.model != proposal.model
  @test res.model.b.model isa MvNormal

  # Both variables
  proposal = symmetric_proposal((; a=BroadcastedNode(:a, rng, KernelNormal, Float32), b=BroadcastedNode(:b, rng, KernelNormal, [0.0f0, 0.0f0], [1.0f0, 1.0f0])), posterior_model)
  res = MCMCDepth.adaptive_mvnormal(rng, proposal, state)
  @test res.model != proposal.model
  @test res.model.a.model isa KernelNormal
  @test res.model.b.model isa MvNormal
end
