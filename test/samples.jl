# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Bijectors
using MCMCDepth
using Random
using Test

@testset "Sample construction & arithmetics" begin
    nta = (; zip((:a, :b), fill(0.5f0, 2))...)
    sa = Sample(nta, 0.0)
    # Broadcasted addition of multiple variables
    ntb = (; zip((:b, :c), fill([1.0, 1.0], 2))...)
    sb = Sample(ntb, 0.0)
    @test logprob(sa) == 0
    @test logprob(sb) == 0

    sum_ab = @inferred sa + sb
    @test logprob(sum_ab) == -Inf
    @test variables(sum_ab).a == 0.5
    @test variables(sum_ab).a isa Float32
    @test variables(sum_ab).b == [1.5, 1.5]
    @test variables(sum_ab).b isa Vector{Float64}

    diff_ab = @inferred sa - sb
    @test logprob(diff_ab) == -Inf
    @test variables(diff_ab).a == 0.5
    @test variables(diff_ab).a isa Float32
    @test variables(diff_ab).b == [-0.5, -0.5]
    @test variables(diff_ab).b isa Vector{Float64}
end

@testset "Sample Bijectors" begin
    rng = Random.default_rng()
    a = BroadcastedNode(:a, rng, KernelExponential, Float16(2.0))
    ta = BroadcastedNode(:a, rng, x -> transformed(KernelExponential(x)), Float16(2.0))
    b = BroadcastedNode(:b, rng, KernelExponential, [2.0f0, 1.0f0, 0.5f0])
    tb = BroadcastedNode(:b, rng, x -> transformed(KernelExponential(x)), [2.0f0, 1.0f0, 0.5f0])
    c = BroadcastedNode(:c, rng, KernelExponential, fill(2.0f0, 2))
    tc = BroadcastedNode(:c, rng, x -> transformed(KernelExponential(x)), fill(2.0f0, 2))

    ab_model = (; a=a, b=b)
    tab_model = (; a=ta, b=tb)

    ab_bijectors = @inferred map(Broadcast.materialize, bijector(ab_model))
    tab_sample = @inferred Sample(rand(tab_model))
    raw_vars = @inferred variables(tab_sample)
    mab_sample, logjac = @inferred to_model_domain(tab_sample, ab_bijectors)
    @test variables(mab_sample).a == invlink(a(), raw_vars.a)
    @test variables(mab_sample).b == invlink(b(), raw_vars.b)
    @test logjac + logdensityof(ab_model, variables(mab_sample)) ≈ logdensityof(tab_model, variables(tab_sample))
    uab_sample = to_unconstrained_domain(mab_sample, ab_bijectors)
    @test mapreduce(≈, &, variables(uab_sample), variables(tab_sample))

    # Partial bijectors
    a_bijector = bijector(a)
    ma_sample, logjac = @inferred to_model_domain(tab_sample, a_bijector)
    @test variables(ma_sample).a == invlink(a(), raw_vars.a)
    @test variables(ma_sample).b == variables(tab_sample).b
    @test logjac == logabsdetjacinv(a_bijector.a, variables(tab_sample).a)
    ua_sample = to_unconstrained_domain(ma_sample, a_bijector)
    @test mapreduce(≈, &, variables(ua_sample), variables(tab_sample))

    # Empty bijectors
    m_sample, logjac = @inferred to_model_domain(tab_sample, (;))
    @test variables(m_sample).a == variables(tab_sample).a
    @test variables(m_sample).b == variables(tab_sample).b
    @test logjac == 0
    u_sample = to_unconstrained_domain(m_sample, (;))
    @test mapreduce(≈, &, variables(u_sample), variables(tab_sample))
end