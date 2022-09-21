# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

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
@test log_prob(sa) == 0
@test log_prob(sb) == 0

sum_ab = @inferred sa + sb
@test log_prob(sum_ab) == -Inf
@test variables(sum_ab).a == 0.5
@test variables(sum_ab).a isa Float32
@test variables(sum_ab).b == [1.5, 1.5]
@test variables(sum_ab).b isa Vector{Float64}

diff_ab = @inferred sa - sb
@test log_prob(diff_ab) == -Inf
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
vars = @inferred variables(tab_sample, ab_bijectors)
@test vars.a == invlink(a_model, raw_vars.a)
@test vars.b == invlink(a_model, raw_vars.b)
w_vars, logjac = @inferred variables_with_logjac(tab_sample, ab_bijectors)
@test vars == w_vars
@test logjac + logdensityof(a_model, vars.a) + logdensityof(b_model, vars.b) ≈ logdensityof(tab_model, tab_sample)

# Partial bijectors
a_bijector = (; a=bijector(a_model))
w_vars, logjac = @inferred variables_with_logjac(tab_sample, a_bijector)
@test w_vars.a == vars.a
@test w_vars.b != vars.b
@test w_vars.b == variables(tab_sample).b
t_vars = variables(tab_sample, a_bijector)
@test w_vars.a == t_vars.a
@test w_vars.b == t_vars.b

# Empty bijectors
w_vars, logjac = @inferred variables_with_logjac(tab_sample, (;))
@test w_vars.a != vars.a
@test w_vars.b != vars.b
@test w_vars.b == variables(tab_sample).b
t_vars = variables(tab_sample, (;))
@test w_vars.a == t_vars.a
@test w_vars.b == t_vars.b
