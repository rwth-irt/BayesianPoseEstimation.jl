# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using MCMCDepth
using Random
using Test

# Prepare a sample
a_model = KernelExponential(2.0)
b_model = ProductDistribution([KernelExponential(2.0f0), KernelExponential(1.0f0), KernelExponential(0.5f0)])
c_model = VectorizedDistribution(fill(KernelExponential(2.0), 2))

abc_model = IndependentModel((; a=a_model, b=b_model, c=c_model))
sample = rand(Random.default_rng(), abc_model)
@test variables(sample).a isa ModelVariable
@test variables(sample).b isa ModelVariable
@test variables(sample).c isa ModelVariable
@test variables(sample).a |> model_value |> size == ()
@test variables(sample).b |> model_value |> size == (3,)
@test variables(sample).c |> model_value |> size == (2,)


# Symmetric proposal
# Only identity bijectors allowed
@test_throws DomainError SymmetricProposal(abc_model)

# Propose single variable
a_normal = KernelNormal(Float64)
a_sym_proposal = SymmetricProposal(IndependentModel((; a=a_normal)))
a_sym_sample = @inferred propose(Random.default_rng(), a_sym_proposal, sample)
@test variables(a_sym_sample).a isa SampleVariable
@test variables(a_sym_sample).b isa ModelVariable
@test variables(a_sym_sample).c isa ModelVariable
@test typeof(bijector(variables(a_sym_sample).a)) == typeof(bijector(variables(sample).a))
@test typeof(bijector(variables(a_sym_sample).b)) == typeof(bijector(variables(sample).b))
@test typeof(bijector(variables(a_sym_sample).c)) == typeof(bijector(variables(sample).c))
@test typeof(model_value(variables(a_sym_sample).a)) == typeof(model_value(variables(sample).a))
@test typeof(model_value(variables(a_sym_sample).b)) == typeof(model_value(variables(sample).b))
@test typeof(model_value(variables(a_sym_sample).c)) == typeof(model_value(variables(sample).c))
@test variables(a_sym_sample).a != variables(sample).a |> model_value
@test variables(a_sym_sample).b |> model_value == variables(sample).b |> model_value
@test variables(a_sym_sample).c |> model_value == variables(sample).c |> model_value
@test variables(a_sym_sample).a |> model_value |> size == ()
@test variables(a_sym_sample).b |> model_value |> size == (3,)
@test variables(a_sym_sample).c |> model_value |> size == (2,)
# Symmetric case: forward equals backward, thus log probability is zero to safe computations
@inferred transition_probability(a_sym_proposal, a_sym_sample, sample)
@test transition_probability(a_sym_proposal, a_sym_sample, sample) == 0

# Propose single variable multiple times
a_sym_sample_2 = @inferred propose(Random.default_rng(), a_sym_proposal, sample, 2)
@test variables(a_sym_sample_2).a isa SampleVariable
@test variables(a_sym_sample_2).b isa ModelVariable
@test variables(a_sym_sample_2).c isa ModelVariable
@test typeof(bijector(variables(a_sym_sample_2).a)) == typeof(bijector(variables(sample).a))
@test typeof(bijector(variables(a_sym_sample_2).b)) == typeof(bijector(variables(sample).b))
@test typeof(bijector(variables(a_sym_sample_2).c)) == typeof(bijector(variables(sample).c))
@test eltype(model_value(variables(a_sym_sample_2).a)) == typeof(model_value(variables(sample).a))
@test typeof(model_value(variables(a_sym_sample_2).b)) == typeof(model_value(variables(sample).b))
@test typeof(model_value(variables(a_sym_sample_2).c)) == typeof(model_value(variables(sample).c))
@test variables(a_sym_sample_2).a |> model_value != variables(sample).a |> model_value
@test variables(a_sym_sample_2).b |> model_value == variables(sample).b |> model_value
@test variables(a_sym_sample_2).c |> model_value == variables(sample).c |> model_value
@test variables(a_sym_sample_2).a |> model_value |> size == (2,)
@test variables(a_sym_sample_2).b |> model_value |> size == (3,)
@test variables(a_sym_sample_2).c |> model_value |> size == (2,)
# Symmetric case: forward equals backward, thus log probability is zero to safe computations
@inferred transition_probability(a_sym_proposal, a_sym_sample_2, sample)
@test transition_probability(a_sym_proposal, a_sym_sample_2, sample) == 0

# Propose multiple variables
b_normal = ProductDistribution(fill(KernelNormal(Float32), 3))
c_normal = VectorizedDistribution(fill(KernelNormal(Float64), 2))

abc_sym_proposal = SymmetricProposal(IndependentModel((; a=a_normal, b=b_normal, c=c_normal)))
# TODO test samples thoroughly for different bijectors and variable types
# WARN does it matter? https://bkamins.github.io/julialang/2021/01/08/typestable.html
abc_sym_sample = @inferred propose(Random.default_rng(), abc_sym_proposal, sample)
@test variables(abc_sym_sample).a |> raw_value |> size == ()
@test variables(abc_sym_sample).a isa SampleVariable
@test variables(abc_sym_sample).b isa SampleVariable
@test variables(abc_sym_sample).c isa SampleVariable
@test typeof(bijector(variables(abc_sym_sample).a)) == typeof(bijector(variables(sample).a))
@test typeof(bijector(variables(abc_sym_sample).b)) == typeof(bijector(variables(sample).b))
@test typeof(bijector(variables(abc_sym_sample).c)) == typeof(bijector(variables(sample).c))
@test typeof(model_value(variables(abc_sym_sample).a)) == typeof(model_value(variables(sample).a))
@test typeof(model_value(variables(abc_sym_sample).b)) == typeof(model_value(variables(sample).b))
@test typeof(model_value(variables(abc_sym_sample).c)) == typeof(model_value(variables(sample).c))
@test variables(abc_sym_sample).a |> model_value != variables(sample).a |> model_value
@test variables(abc_sym_sample).b |> model_value != variables(sample).b |> model_value
@test variables(abc_sym_sample).c |> model_value != variables(sample).c |> model_value
@test variables(abc_sym_sample).a |> model_value |> size == ()
@test variables(abc_sym_sample).b |> model_value |> size == (3,)
@test variables(abc_sym_sample).c |> model_value |> size == (2,)
# Symmetric case: forward equals backward, thus log probability is zero to safe computations
@inferred transition_probability(abc_sym_proposal, abc_sym_sample, sample)
@test transition_probability(abc_sym_proposal, abc_sym_sample, sample) == 0

# Propose multiple variables multiple times
abc_sym_sample_2 = @inferred propose(Random.default_rng(), abc_sym_proposal, sample, 2)
@test variables(abc_sym_sample_2).a isa SampleVariable
@test variables(abc_sym_sample_2).b isa SampleVariable
@test variables(abc_sym_sample_2).c isa SampleVariable
@test typeof(bijector(variables(abc_sym_sample_2).a)) == typeof(bijector(variables(sample).a))
@test typeof(bijector(variables(abc_sym_sample_2).b)) == typeof(bijector(variables(sample).b))
@test typeof(bijector(variables(abc_sym_sample_2).c)) == typeof(bijector(variables(sample).c))
@test eltype(model_value(variables(abc_sym_sample_2).a)) == typeof(model_value(variables(sample).a))
@test eltype(model_value(variables(abc_sym_sample_2).b)) == eltype(model_value(variables(sample).b))
@test eltype(model_value(variables(abc_sym_sample_2).c)) == eltype(model_value(variables(sample).c))
@test variables(abc_sym_sample_2).a |> model_value != variables(sample).a |> model_value
@test variables(abc_sym_sample_2).b |> model_value != variables(sample).b |> model_value
@test variables(abc_sym_sample_2).c |> model_value != variables(sample).c |> model_value
@test variables(abc_sym_sample_2).a |> model_value |> size == (2,)
@test variables(abc_sym_sample_2).b |> model_value |> size == (3, 2)
@test variables(abc_sym_sample_2).c |> model_value |> size == (2, 2)
# Symmetric case: forward equals backward, thus log probability is zero to safe computations
@inferred transition_probability(abc_sym_proposal, abc_sym_sample_2, sample)
@test transition_probability(abc_sym_proposal, abc_sym_sample_2, sample) == 0


# Independent proposal

# Propose single variable
a_ind_proposal = IndependentProposal(IndependentModel((; a=a_model)))
a_ind_sample = @inferred propose(Random.default_rng(), a_ind_proposal, MCMCDepth.to_sample_variables(sample))
@test variables(a_ind_sample).a isa ModelVariable
@test variables(a_ind_sample).b isa SampleVariable
@test variables(a_ind_sample).c isa SampleVariable
@test typeof(bijector(variables(a_ind_sample).a)) == typeof(bijector(variables(sample).a))
@test typeof(bijector(variables(a_ind_sample).b)) == typeof(bijector(variables(sample).b))
@test typeof(bijector(variables(a_ind_sample).c)) == typeof(bijector(variables(sample).c))
@test typeof(model_value(variables(a_ind_sample).a)) == typeof(model_value(variables(sample).a))
@test typeof(model_value(variables(a_ind_sample).b)) == typeof(model_value(variables(sample).b))
@test typeof(model_value(variables(a_ind_sample).c)) == typeof(model_value(variables(sample).c))
@test variables(a_ind_sample).a |> model_value != variables(sample).a |> model_value
@test variables(a_ind_sample).b |> model_value ≈ variables(sample).b |> model_value
@test variables(a_ind_sample).c |> model_value ≈ variables(sample).c |> model_value
@test variables(a_ind_sample).a |> model_value |> size == ()
@test variables(a_ind_sample).b |> model_value |> size == (3,)
@test variables(a_ind_sample).c |> model_value |> size == (2,)
# Logdensity of independent components is the sum of all the components
@inferred transition_probability(a_ind_proposal, a_ind_sample, sample)
@test transition_probability(a_ind_proposal, a_ind_sample, sample) == logdensityof(a_model, variables(a_ind_sample).a)

# Propose single variable multiple times
a_ind_sample_2 = @inferred propose(Random.default_rng(), a_ind_proposal, MCMCDepth.to_sample_variables(sample), 2)
@test variables(a_ind_sample_2).a isa ModelVariable
@test variables(a_ind_sample_2).b isa SampleVariable
@test variables(a_ind_sample_2).c isa SampleVariable
@test typeof(bijector(variables(a_ind_sample_2).a)) == typeof(bijector(variables(sample).a))
@test typeof(bijector(variables(a_ind_sample_2).b)) == typeof(bijector(variables(sample).b))
@test typeof(bijector(variables(a_ind_sample_2).c)) == typeof(bijector(variables(sample).c))
@test eltype(model_value(variables(a_ind_sample_2).a)) == typeof(model_value(variables(sample).a))
@test typeof(model_value(variables(a_ind_sample_2).b)) == typeof(model_value(variables(sample).b))
@test typeof(model_value(variables(a_ind_sample_2).c)) == typeof(model_value(variables(sample).c))
@test variables(a_ind_sample_2).a |> model_value != variables(sample).a |> model_value
@test variables(a_ind_sample_2).b |> model_value ≈ variables(sample).b |> model_value
@test variables(a_ind_sample_2).c |> model_value ≈ variables(sample).c |> model_value
@test variables(a_ind_sample_2).a |> model_value |> size == (2,)
@test variables(a_ind_sample_2).b |> model_value |> size == (3,)
@test variables(a_ind_sample_2).c |> model_value |> size == (2,)
# Logdensity of independent components is the sum of all the components
@inferred transition_probability(a_ind_proposal, a_ind_sample_2, sample)
@test transition_probability(a_ind_proposal, a_ind_sample_2, sample) == logdensityof(a_model, variables(a_ind_sample_2).a)

# Propose multiple variables
abc_ind_proposal = IndependentProposal(IndependentModel((; a=a_model, b=b_model, c=c_model)))
abc_ind_sample = @inferred propose(Random.default_rng(), abc_ind_proposal, sample)
@test variables(abc_ind_sample).a |> raw_value |> size == ()
@test variables(abc_ind_sample).a isa ModelVariable
@test variables(abc_ind_sample).b isa ModelVariable
@test variables(abc_ind_sample).c isa ModelVariable
@test typeof(bijector(variables(abc_ind_sample).a)) == typeof(bijector(variables(sample).a))
@test typeof(bijector(variables(abc_ind_sample).b)) == typeof(bijector(variables(sample).b))
@test typeof(bijector(variables(abc_ind_sample).c)) == typeof(bijector(variables(sample).c))
@test typeof(model_value(variables(abc_ind_sample).a)) == typeof(model_value(variables(sample).a))
@test typeof(model_value(variables(abc_ind_sample).b)) == typeof(model_value(variables(sample).b))
@test typeof(model_value(variables(abc_ind_sample).c)) == typeof(model_value(variables(sample).c))
@test variables(abc_ind_sample).a |> model_value != variables(sample).a |> model_value
@test variables(abc_ind_sample).b |> model_value != variables(sample).b |> model_value
@test variables(abc_ind_sample).c |> model_value != variables(sample).c |> model_value
@test variables(abc_ind_sample).a |> model_value |> size == ()
@test variables(abc_ind_sample).b |> model_value |> size == (3,)
@test variables(abc_ind_sample).c |> model_value |> size == (2,)
# Logdensity of independent components is the sum of all the components
@inferred transition_probability(abc_ind_proposal, abc_ind_sample, sample)
@test transition_probability(abc_ind_proposal, abc_ind_sample, sample) == logdensityof(a_model, variables(abc_ind_sample).a) .+ logdensityof(b_model, variables(abc_ind_sample).b) .+ logdensityof(c_model, variables(abc_ind_sample).c)

# Propose multiple variables multiple times
abc_ind_sample_2 = @inferred propose(Random.default_rng(), abc_ind_proposal, sample, 2)
@test variables(abc_ind_sample_2).a isa ModelVariable
@test variables(abc_ind_sample_2).b isa ModelVariable
@test variables(abc_ind_sample_2).c isa ModelVariable
@test typeof(bijector(variables(abc_ind_sample_2).a)) == typeof(bijector(variables(sample).a))
@test typeof(bijector(variables(abc_ind_sample_2).b)) == typeof(bijector(variables(sample).b))
@test typeof(bijector(variables(abc_ind_sample_2).c)) == typeof(bijector(variables(sample).c))
@test eltype(model_value(variables(abc_ind_sample_2).a)) == typeof(model_value(variables(sample).a))
@test eltype(model_value(variables(abc_ind_sample_2).b)) == eltype(model_value(variables(sample).b))
@test eltype(model_value(variables(abc_ind_sample_2).c)) == eltype(model_value(variables(sample).c))
@test variables(abc_ind_sample_2).a |> model_value != variables(sample).a |> model_value
@test variables(abc_ind_sample_2).b |> model_value != variables(sample).b |> model_value
@test variables(abc_ind_sample_2).c |> model_value != variables(sample).c |> model_value
@test variables(abc_ind_sample_2).a |> model_value |> size == (2,)
@test variables(abc_ind_sample_2).b |> model_value |> size == (3, 2)
@test variables(abc_ind_sample_2).c |> model_value |> size == (2, 2)
# Logdensity of independent components is the sum of all the components
@inferred transition_probability(abc_ind_proposal, abc_ind_sample_2, sample)
@test transition_probability(abc_ind_proposal, abc_ind_sample_2, sample) == logdensityof(a_model, variables(abc_ind_sample_2).a) .+ logdensityof(b_model, variables(abc_ind_sample_2).b) .+ logdensityof(c_model, variables(abc_ind_sample_2).c)


# Gibbs

# Simple function that only changes the type of the variable which allows to test the merge inside the Gibbs proposal
a_gibbs_fn(sample) = Sample((; a=SampleVariable(variables(sample).a)), 42.0)
a_gibbs_proposal = GibbsProposal(a_gibbs_fn)
a_gibbs_sample = @inferred propose(Random.default_rng(), a_gibbs_proposal, sample)
@test variables(a_gibbs_sample).a isa SampleVariable
@test variables(a_gibbs_sample).b isa ModelVariable
@test variables(a_gibbs_sample).c isa ModelVariable
@test typeof(bijector(variables(a_gibbs_sample).a)) == typeof(bijector(variables(sample).a))
@test typeof(bijector(variables(a_gibbs_sample).b)) == typeof(bijector(variables(sample).b))
@test typeof(bijector(variables(a_gibbs_sample).c)) == typeof(bijector(variables(sample).c))
@test typeof(model_value(variables(a_gibbs_sample).a)) == typeof(model_value(variables(sample).a))
@test typeof(model_value(variables(a_gibbs_sample).b)) == typeof(model_value(variables(sample).b))
@test typeof(model_value(variables(a_gibbs_sample).c)) == typeof(model_value(variables(sample).c))
@test variables(a_gibbs_sample).a |> model_value == variables(sample).a |> model_value
@test variables(a_gibbs_sample).b |> model_value == variables(sample).b |> model_value
@test variables(a_gibbs_sample).c |> model_value == variables(sample).c |> model_value
@test variables(a_gibbs_sample).a |> model_value |> size == ()
@test variables(a_gibbs_sample).b |> model_value |> size == (3,)
@test variables(a_gibbs_sample).c |> model_value |> size == (2,)
# Logdensity of independent components is the sum of all the components
@inferred transition_probability(a_gibbs_proposal, a_gibbs_sample, sample)
@test isinf(transition_probability(a_gibbs_proposal, a_gibbs_sample, sample))