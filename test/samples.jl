# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Distributions
using MCMCDepth
using Random
using Test

a_model = KernelExponential(Float16(2.0))
b_model = BroadcastedDistribution(Exponential, [2.0f0, 1.0f0, 0.5f0])
c_model = BroadcastedDistribution(KernelExponential, fill(2.0f0, 2))

abc_model = IndependentModel((; a=a_model, b=b_model, c=c_model))

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
