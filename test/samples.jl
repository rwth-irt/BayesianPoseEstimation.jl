# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Bijectors
using Distributions
using MCMCDepth
using Random
using Test

a_model = KernelExponential(Float16(2.0))
b_model = ProductBroadcastedDistribution(Exponential, [2.0f0, 1.0f0, 0.5f0])
c_model = ProductBroadcastedDistribution(KernelExponential, fill(2.0f0, 2))

# Sample
nta = (; zip((:a, :b), fill(0.5f0, 2))...)
# Broadcasted addition of multiple variables
ntb = (; zip((:b, :c), fill([1.0, 1.0], 2))...)
sa = Sample(nta, 0.0)
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

# Bijectors
ab_model = IndependentModel((; a=a_model, b=b_model))
ab_bijectors = @inferred map(Broadcast.materialize, bijector(ab_model))
tab_model = @inferred transformed(ab_model)
tab_sample = rand(tab_model)
raw_vars = @inferred variables(tab_sample)
mab_sample, logjac = @inferred to_model_domain(tab_sample, ab_bijectors)
@test variables(mab_sample).a == invlink(a_model, raw_vars.a)
@test variables(mab_sample).b == invlink(a_model, raw_vars.b)
@test logjac + logdensityof(ab_model, mab_sample) ≈ logdensityof(tab_model, tab_sample)
uab_sample = to_unconstrained_domain(mab_sample, ab_bijectors)
@test mapreduce(≈, &, variables(uab_sample), variables(tab_sample))

# Partial bijectors
a_bijector = (; a=bijector(a_model))
ma_sample, logjac = @inferred to_model_domain(tab_sample, a_bijector)
@test variables(ma_sample).a == invlink(a_model, raw_vars.a)
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
