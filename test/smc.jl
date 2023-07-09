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
b = BroadcastedNode(:b, rng, KernelExponential, 1.0f0)
c = BroadcastedNode(:c, rng, KernelNormal, (a, b))
# TODO one has to know in advance that SMC will add the second dimension and also care for the model dimensions
modifier_fn(args...) = SumLogdensityModifier((2,))
c_mod = ModifierNode(c, rng, modifier_fn)
model = sequentialize(c_mod)

# TODO adding the dimension is not obvious to force broadcasting over the correct one
data = (; c=rand(c_mod, 1, 50).c)
posterior_model = PosteriorModel(c_mod, data)
posterior_model.likelihood

@testset "Type stable ModifierNode" begin
    # Had issues with model(::ModifierNode) being unstable when MCMCDepth was loaded.
    sample = @inferred rand(posterior_model)
    ℓ = @inferred logdensityof(posterior_model, sample)
    @test ℓ isa AbstractVector{Float32}
    @test length(ℓ) == 1
end

proposal_model = (; a=SimpleNode(:a, rng, KernelNormal, Float32), b=SimpleNode(:b, rng, KernelNormal, Float32))
proposal = symmetric_proposal(proposal_model, posterior_model)

# @testset "SMC forward kernel" begin
kernel = ForwardProposalKernel(proposal)
n_steps = 42
n_particles = 6
smc = SequentialMonteCarlo(kernel, LinearSchedule(n_steps), n_particles, log(0.5 * n_particles))
sample, state = AbstractMCMC.step(rng, posterior_model, smc)
# end