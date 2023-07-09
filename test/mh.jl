# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using KernelDistributions
using MCMCDepth
using Random
using Test

rng = Random.default_rng()

@testset "MH accepatance ratio" begin
    p = BroadcastedNode(:a, rng, KernelNormal, 1.0, 2.0)
    q = BroadcastedNode(:a, rng, KernelNormal, 0.0, 1.0)
    proposal = symmetric_proposal(q, p)
    nt1 = rand(q)
    s1 = Sample(nt1, logdensityof(p, nt1))
    nt2 = propose(proposal, s1) |> variables
    s2 = Sample(nt2, logdensityof(p, nt2))
    α = @inferred MCMCDepth.acceptance_ratio(proposal, s2, s1)

    # Some fake logdensities to check values
    s1 = Sample(nt1, 1)
    s2 = Sample(nt2, 1)
    α = @inferred MCMCDepth.acceptance_ratio(proposal, s2, s1)
    @test α === 0
    s2 = Sample(nt2, 2)
    α = @inferred MCMCDepth.acceptance_ratio(proposal, s2, s1)
    @test α === 1
    s2 = Sample(nt2, [1, 2])
    α = @inferred MCMCDepth.acceptance_ratio(proposal, s2, s1)
    @test α == [0, 1]
end

@testset "MH Should reject" begin
    @test @inferred MCMCDepth.should_reject(rng, 0) == false
    @test @inferred MCMCDepth.should_reject(rng, 0.1) == false
    @test @inferred MCMCDepth.should_reject(rng, -Inf) == true
    α = @inferred sum([MCMCDepth.should_reject(rng, log(0.5)) for _ in 1:100_000]) / 100_000
    @test isapprox(α, 0.5; atol=0.01)

    @test @inferred MCMCDepth.should_reject(rng, [0, 0.1]) == [false, false]
    @test @inferred MCMCDepth.should_reject(rng, [0, -Inf]) == [false, true]

    # Reject, scalar should just return the previous or proposed
    @test @inferred MCMCDepth.reject_barrier(true, 1, 2) == 2
    @test @inferred MCMCDepth.reject_barrier(false, [1], [2]) == [1]
    @test @inferred MCMCDepth.reject_barrier(false, :proposed, :previous) == :proposed

    # Reject, vectorized should select from the arrays
    previous = Sample((; a=fill(1, 2, 3)))
    proposed = Sample((; a=fill(2, 2, 3)), [1.0, 2.0, 3.0])
    rejected = [true, false, false]
    result = @inferred MCMCDepth.reject_barrier(rejected, proposed, previous)
    @test variables(result) == (; a=[1 2 2; 1 2 2])
    @test logprob(result) == [-Inf, 2.0, 3.0]

    # Scalar previous
    previous = Sample((; a=1))
    proposed = Sample((; a=fill(2, 2, 3)), [1.0, 2.0, 3.0])
    rejected = [true, false, false]
    result = @inferred MCMCDepth.reject_barrier(rejected, proposed, previous)
    @test variables(result) == (; a=[1 2 2; 1 2 2])
    @test logprob(result) == [-Inf, 2.0, 3.0]

    # Smaller previous
    previous = Sample((; a=fill(1, 2)))
    proposed = Sample((; a=fill(2, 2, 3)), [1.0, 2.0, 3.0])
    rejected = [true, false, false]
    result = @inferred MCMCDepth.reject_barrier(rejected, proposed, previous)
    @test variables(result) == (; a=[1 2 2; 1 2 2])
    @test logprob(result) == [-Inf, 2.0, 3.0]

    # Larger previous
    previous = Sample((; a=fill(1, 2, 2, 3)))
    proposed = Sample((; a=fill(2, 2, 3)), [1.0, 2.0, 3.0])
    rejected = [true, false, false]
    @test_throws ArgumentError MCMCDepth.reject_barrier(rejected, proposed, previous)
end