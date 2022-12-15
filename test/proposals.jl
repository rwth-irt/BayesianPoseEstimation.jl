# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using Distributions
using MCMCDepth
using Random
using Test

# Prepare a sample
rng = Random.default_rng()
a = BroadcastedNode(:a, rng, KernelExponential, Float16(2.0))
b = BroadcastedNode(:b, rng, KernelExponential, [2.0f0, 1.0f0, 0.5f0])
rand(b)
c = BroadcastedNode(:c, rng, KernelNormal, (; a=a, b=b))

s = Sample(rand(c))

@test variables(s).a |> size == ()
@test variables(s).b |> size == (3,)
@test variables(s).c |> size == (3,)

# Additive Proposal
# Propose single variable
a_normal = BroadcastedNode(:a, rng, KernelNormal, 0, 1.0)
a_add_proposal = additive_proposal(a_normal, c)
a_add_sample = @inferred propose(a_add_proposal, s)
@test variables(a_add_sample).a isa Float64
@test typeof(variables(a_add_sample).b) == typeof(variables(s).b)
@test typeof(variables(a_add_sample).c) == typeof(variables(s).c)
@test variables(a_add_sample).a != variables(s).a
@test variables(a_add_sample).b ≈ variables(s).b
@test variables(a_add_sample).c == variables(s).c
@test variables(a_add_sample).a |> size == ()
@test variables(a_add_sample).b |> size == (3,)
@test variables(a_add_sample).c |> size == (3,)
# Forwards should equal backwards for additive normally distributed proposal
ℓ = @inferred transition_probability(a_add_proposal, a_add_sample, s)
@test !iszero(ℓ)
@test ℓ == transition_probability(a_add_proposal, s, a_add_sample)

# Propose single variable multiple times
a_add_sample_2 = @inferred propose(a_add_proposal, s, 2)
@test variables(a_add_sample_2).a isa Vector{Float64}
@test typeof(variables(a_add_sample_2).b) == typeof(variables(s).b)
@test typeof(variables(a_add_sample_2).c) == typeof(variables(s).c)
@test variables(a_add_sample_2).a != variables(s).a
@test variables(a_add_sample_2).b ≈ variables(s).b
@test variables(a_add_sample_2).c == variables(s).c
@test variables(a_add_sample_2).a |> size == (2,)
@test variables(a_add_sample_2).b |> size == (3,)
@test variables(a_add_sample_2).c |> size == (3,)
# Forwards should equal backwards for additive normally distributed proposal
ℓ = @inferred transition_probability(a_add_proposal, a_add_sample_2, s)
@test !iszero(ℓ)
@test ℓ == transition_probability(a_add_proposal, s, a_add_sample_2)

# Propose multiple variables
b_normal = BroadcastedNode(:b, rng, KernelNormal, fill(0.0f0, 3), fill(1.0f0, 3))
ab_add_proposal = additive_proposal((; a=a_normal, b=b_normal), c)
# WARN does it matter? https://bkamins.github.io/julialang/2021/01/08/typestable.html
ab_add_sample = @inferred propose(ab_add_proposal, s)
@test variables(ab_add_sample).a |> size == ()
@test variables(ab_add_sample).a isa Float64
@test variables(ab_add_sample).b isa Vector{Float32}
@test variables(ab_add_sample).c isa Vector{Float32}
@test variables(ab_add_sample).a != variables(s).a
@test variables(ab_add_sample).b != variables(s).b
@test variables(ab_add_sample).c == variables(s).c
@test variables(ab_add_sample).a |> size == ()
@test variables(ab_add_sample).b |> size == (3,)
@test variables(ab_add_sample).c |> size == (3,)
# Forwards should equal backwards for additive normally distributed proposal
ℓ = @inferred transition_probability(ab_add_proposal, ab_add_sample, s)
@test !iszero(ℓ)
@test ℓ == transition_probability(ab_add_proposal, s, ab_add_sample)

# Propose multiple variables multiple times
ab_add_sample_2 = @inferred propose(ab_add_proposal, s, 2)
@test variables(ab_add_sample_2).a isa Vector{Float64}
@test variables(ab_add_sample_2).b isa Matrix{Float32}
@test variables(ab_add_sample_2).c isa Vector{Float32}
@test variables(ab_add_sample_2).a != variables(s).a
@test variables(ab_add_sample_2).b != variables(s).b
@test variables(ab_add_sample_2).c == variables(s).c
@test variables(ab_add_sample_2).a |> size == (2,)
@test variables(ab_add_sample_2).b |> size == (3, 2)
@test variables(ab_add_sample_2).c |> size == (3,)
# Forwards should equal backwards for additive normally distributed proposal
ℓ = @inferred transition_probability(ab_add_proposal, ab_add_sample_2, s)
@test !iszero(ℓ)
@test ℓ == transition_probability(ab_add_proposal, s, ab_add_sample_2)

# Symmetric Proposal
# Propose single variable
a_normal = BroadcastedNode(:a, rng, KernelNormal, 0, 1.0)
a_sym_proposal = symmetric_proposal((; a=a_normal), c)
a_sym_sample = @inferred propose(a_sym_proposal, s)
@test variables(a_sym_sample).a isa Float64
@test typeof(variables(a_sym_sample).b) == typeof(variables(s).b)
@test typeof(variables(a_sym_sample).c) == typeof(variables(s).c)
@test variables(a_sym_sample).a != variables(s).a
@test variables(a_sym_sample).b ≈ variables(s).b
@test variables(a_sym_sample).c == variables(s).c
@test variables(a_sym_sample).a |> size == ()
@test variables(a_sym_sample).b |> size == (3,)
@test variables(a_sym_sample).c |> size == (3,)
# Symmetric case: forward equals backward, thus log probability is zero to safe computations
@inferred transition_probability(a_sym_proposal, a_sym_sample, s)
@test transition_probability(a_sym_proposal, a_sym_sample, s) == 0

# Propose single variable multiple times
a_sym_sample_2 = @inferred propose(a_sym_proposal, s, 2)
@test variables(a_sym_sample_2).a isa Vector{Float64}
@test typeof(variables(a_sym_sample_2).b) == typeof(variables(s).b)
@test typeof(variables(a_sym_sample_2).c) == typeof(variables(s).c)
@test variables(a_sym_sample_2).a != variables(s).a
@test variables(a_sym_sample_2).b ≈ variables(s).b
@test variables(a_sym_sample_2).c == variables(s).c
@test variables(a_sym_sample_2).a |> size == (2,)
@test variables(a_sym_sample_2).b |> size == (3,)
@test variables(a_sym_sample_2).c |> size == (3,)
# Symmetric case: forward equals backward, thus log probability is zero to safe computations
@inferred transition_probability(a_sym_proposal, a_sym_sample_2, s)
@test transition_probability(a_sym_proposal, a_sym_sample_2, s) == 0

# Propose multiple variables
b_normal = BroadcastedNode(:b, rng, KernelNormal, fill(0.0f0, 3), fill(1.0f0, 3))
ab_sym_proposal = symmetric_proposal((; a=a_normal, b=b_normal), c)
# WARN does it matter? https://bkamins.github.io/julialang/2021/01/08/typestable.html
ab_sym_sample = @inferred propose(ab_sym_proposal, s)
@test variables(ab_sym_sample).a |> size == ()
@test variables(ab_sym_sample).a isa Float64
@test variables(ab_sym_sample).b isa Vector{Float32}
@test variables(ab_sym_sample).c isa Vector{Float32}
@test variables(ab_sym_sample).a != variables(s).a
@test variables(ab_sym_sample).b != variables(s).b
@test variables(ab_sym_sample).c == variables(s).c
@test variables(ab_sym_sample).a |> size == ()
@test variables(ab_sym_sample).b |> size == (3,)
@test variables(ab_sym_sample).c |> size == (3,)
# Symmetric case: forward equals backward, thus log probability is zero to safe computations
@inferred transition_probability(ab_sym_proposal, ab_sym_sample, s)
@test transition_probability(ab_sym_proposal, ab_sym_sample, s) == 0

# Propose multiple variables multiple times
ab_sym_sample_2 = @inferred propose(ab_sym_proposal, s, 2)
@test variables(ab_sym_sample_2).a isa Vector{Float64}
@test variables(ab_sym_sample_2).b isa Matrix{Float32}
@test variables(ab_sym_sample_2).c isa Vector{Float32}
@test variables(ab_sym_sample_2).a != variables(s).a
@test variables(ab_sym_sample_2).b != variables(s).b
@test variables(ab_sym_sample_2).c == variables(s).c
@test variables(ab_sym_sample_2).a |> size == (2,)
@test variables(ab_sym_sample_2).b |> size == (3, 2)
@test variables(ab_sym_sample_2).c |> size == (3,)
# Symmetric case: forward equals backward, thus log probability is zero to safe computations
@inferred transition_probability(ab_sym_proposal, ab_sym_sample_2, s)
@test transition_probability(ab_sym_proposal, ab_sym_sample_2, s) == 0

# Independent proposal

# Propose single variable
a_ind_proposal = independent_proposal((; a=a), c)
a_ind_sample = @inferred propose(a_ind_proposal, s)
@test variables(a_ind_sample).a isa Float16
@test typeof(variables(a_ind_sample).b) == typeof(variables(s).b)
@test typeof(variables(a_ind_sample).c) == typeof(variables(s).c)
@test variables(a_ind_sample).a != variables(s).a
@test variables(a_ind_sample).b ≈ variables(s).b
@test variables(a_ind_sample).c == variables(s).c
@test variables(a_ind_sample).a |> size == ()
@test variables(a_ind_sample).b |> size == (3,)
@test variables(a_ind_sample).c |> size == (3,)
# Logdensity of independent components is the sum of all the components
ℓ = @inferred transition_probability(a_ind_proposal, a_ind_sample, s)
@test ℓ == logdensityof(transformed(a()), variables(a_ind_sample).a)

# Propose single variable multiple times
a_ind_sample_2 = @inferred propose(a_ind_proposal, s, 2)
@test variables(a_ind_sample_2).a isa Vector{Float16}
@test variables(a_ind_sample_2).b isa typeof(variables(s).b)
@test variables(a_ind_sample_2).c isa typeof(variables(s).c)
@test variables(a_ind_sample_2).a != variables(s).a
@test variables(a_ind_sample_2).b ≈ variables(s).b
@test variables(a_ind_sample_2).c == variables(s).c
@test variables(a_ind_sample_2).a |> size == (2,)
@test variables(a_ind_sample_2).b |> size == (3,)
@test variables(a_ind_sample_2).c |> size == (3,)
# Logdensity of independent components is the sum of all the components
@inferred transition_probability(a_ind_proposal, a_ind_sample_2, s)
@test transition_probability(a_ind_proposal, a_ind_sample_2, s) == logdensityof.(transformed(a()), variables(a_ind_sample_2).a)

# Propose multiple variables
ab_ind_proposal = independent_proposal((; a=a, b=b), c)
ab_ind_sample = @inferred propose(ab_ind_proposal, s)
@test variables(ab_ind_sample).a |> size == ()
@test typeof(variables(ab_ind_sample).a) == typeof(variables(s).a)
@test typeof(variables(ab_ind_sample).b) == typeof(variables(s).b)
@test typeof(variables(ab_ind_sample).c) == typeof(variables(s).c)
@test variables(ab_ind_sample).a != variables(s).a
@test variables(ab_ind_sample).b != variables(s).b
@test variables(ab_ind_sample).c == variables(s).c
@test variables(ab_ind_sample).a |> size == ()
@test variables(ab_ind_sample).b |> size == (3,)
@test variables(ab_ind_sample).c |> size == (3,)
# Logdensity of independent components is the sum of all the components
@inferred transition_probability(ab_ind_proposal, ab_ind_sample, s)
abc_ind_model_sample, _ = to_model_domain(ab_ind_sample, bijector(c))
@test transition_probability(ab_ind_proposal, ab_ind_sample, s) ≈ logdensityof(transformed(a()), variables(ab_ind_sample).a) .+ logdensityof(transformed(b()), variables(ab_ind_sample).b)
# Propose multiple variables multiple times
ab_ind_sample_2 = @inferred propose(ab_ind_proposal, s, 2)
@test eltype(variables(ab_ind_sample_2).a) == typeof(variables(s).a)
@test eltype(variables(ab_ind_sample_2).b) == eltype(variables(s).b)
@test eltype(variables(ab_ind_sample_2).c) == eltype(variables(s).c)
@test variables(ab_ind_sample_2).a != variables(s).a
@test variables(ab_ind_sample_2).b != variables(s).b
@test variables(ab_ind_sample_2).c == variables(s).c
@test variables(ab_ind_sample_2).a |> size == (2,)
@test variables(ab_ind_sample_2).b |> size == (3, 2)
@test variables(ab_ind_sample_2).c |> size == (3,)
# Logdensity of independent components is the sum of all the components
@inferred transition_probability(ab_ind_proposal, ab_ind_sample_2, s)
@test transition_probability(ab_ind_proposal, ab_ind_sample, s) ≈ logdensityof(transformed(a()), variables(ab_ind_sample).a) .+ logdensityof(transformed(b()), variables(ab_ind_sample).b)

# evaluation of DeterministicNode
a = BroadcastedNode(:a, rng, KernelExponential, [2.0f0, 1.0f0, 0.5f0])
fn(x) = 2 * x
b = DeterministicNode(:b, fn, (; a=a))
c = BroadcastedNode(:c, rng, KernelExponential, (; b=b))
s = rand(c) |> Sample

# evaluation should update b
a_sym = symmetric_proposal(BroadcastedNode(:a, rng, KernelNormal, 0, fill(1.0f0, 3)), c)
proposed = @inferred propose(a_sym, s, 2)
@test variables(proposed).a != variables(s).a
@test variables(proposed).a isa Array{Float32,2}
@test variables(proposed).b != variables(s).b
@test variables(proposed).b isa Array{Float32,2}
@test variables(proposed).c ≈ variables(s).c
@test variables(proposed).c isa Array{Float32,1}

# evaluation should not update b
c_sym = symmetric_proposal(BroadcastedNode(:c, rng, KernelNormal, 0, fill(1.0f0, 3)), c)
proposed = @inferred propose(c_sym, s, 2)
@test variables(proposed).a ≈ variables(s).a
@test variables(proposed).a isa Array{Float32,1}
@test variables(proposed).b == variables(s).b
@test variables(proposed).b isa Array{Float32,1}
@test variables(proposed).c != variables(s).c
@test variables(proposed).c isa Array{Float32,2}

# evaluation should update b
ac_sym = symmetric_proposal((; a=BroadcastedNode(:a, rng, KernelNormal, 0, 1fill(1.0f0, 3)), c=BroadcastedNode(:c, rng, KernelNormal, 0, fill(1.0f0, 3))), c)
proposed = @inferred propose(ac_sym, s, 2)
@test variables(proposed).a != variables(s).a
@test variables(proposed).a isa Array{Float32,2}
@test variables(proposed).b != variables(s).b
@test variables(proposed).b isa Array{Float32,2}
@test variables(proposed).c != variables(s).c
@test variables(proposed).c isa Array{Float32,2}

# evaluation should update b
a_ind = symmetric_proposal(BroadcastedNode(:a, rng, KernelExponential, 1.0f0), c)
proposed = @inferred propose(a_ind, s)
@test variables(proposed).a != variables(s).a
@test variables(proposed).a isa Vector{Float32}
@test variables(proposed).b != variables(s).b
@test variables(proposed).b isa Vector{Float32}
@test variables(proposed).c ≈ variables(s).c
@test variables(proposed).c isa Array{Float32,1}

# evaluation should not update b
c_ind = independent_proposal(BroadcastedNode(:c, rng, KernelExponential, 1.0f0), c)
proposed = @inferred propose(c_ind, s)
@test variables(proposed).a ≈ variables(s).a
@test variables(proposed).a isa Array{Float32,1}
@test variables(proposed).b == variables(s).b
@test variables(proposed).b isa Array{Float32,1}
@test variables(proposed).c != variables(s).c
@test variables(proposed).c isa Float32

# evaluation should update b
ac_ind = independent_proposal((; a=BroadcastedNode(:a, rng, KernelExponential, 1.0f0), c=BroadcastedNode(:c, rng, KernelExponential, 1.0f0)), c)
proposed = @inferred propose(ac_ind, s)
@test variables(proposed).a != variables(s).a
@test variables(proposed).a isa Float32
@test variables(proposed).b != variables(s).b
@test variables(proposed).b isa Float32
@test variables(proposed).c != variables(s).c
@test variables(proposed).c isa Float32

# TODO move to sampler tests # Gibbs
# # Simple function that only changes the type of the variable which allows to test the merge inside the Gibbs proposal
# a_gibbs_fn(sample) = Sample((; a=42.0f0), 42.0)
# a_gibbs_proposal = GibbsProposal(a_gibbs_fn)
# a_gibbs_sample = @inferred propose(Random.default_rng(), a_gibbs_proposal, sample)
# @test variables(a_gibbs_sample).a == 42
# @test typeof(variables(a_gibbs_sample).a) == Float32
# @test typeof(variables(a_gibbs_sample).b) == typeof(variables(sample).b)
# @test typeof(variables(a_gibbs_sample).c) == typeof(variables(sample).c)
# @test variables(a_gibbs_sample).a == 42.0f0
# @test variables(a_gibbs_sample).b == variables(sample).b
# @test variables(a_gibbs_sample).c == variables(sample).c
# @test variables(a_gibbs_sample).a |> size == ()
# @test variables(a_gibbs_sample).b |> size == (3,)
# @test variables(a_gibbs_sample).c |> size == (2,)
# # Logdensity of independent components is the sum of all the components
# @inferred transition_probability(a_gibbs_proposal, a_gibbs_sample, sample)
# @test isinf(transition_probability(a_gibbs_proposal, a_gibbs_sample, sample))
