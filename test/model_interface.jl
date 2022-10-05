# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 


# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using Accessors
using Bijectors
using Distributions
using MCMCDepth
using Random
using Test

a_model = KernelExponential(Float16(2.0))
b_model = ProductBroadcastedDistribution(Exponential, [2.0f0, 1.0f0, 0.5f0])
c_model = ProductBroadcastedDistribution(KernelExponential, fill(2.0f0, 2))

ab_model = IndependentModel((; a=a_model, b=b_model))
ac_model = IndependentModel((; a=a_model, c=c_model))
bc_model = IndependentModel((; a=a_model, c=c_model))
abc_model = IndependentModel((; a=a_model, b=b_model, c=c_model))

# IndependentModel
s = @inferred rand(Random.default_rng(), abc_model)
@test variables(s).a isa Float16
@test variables(s).b isa Vector{Float32}
@test variables(s).c isa Vector{Float32}
@inferred logdensityof(IndependentModel((; a=a_model)), s)
@inferred logdensityof(ab_model, s)
@inferred logdensityof(ac_model, s)
@inferred logdensityof(bc_model, s)
@inferred logdensityof(abc_model, s)
s = @inferred rand(Random.default_rng(), abc_model, 3)
@test variables(s).a isa Vector{Float16}
@test variables(s).b isa Matrix{Float32}
@test variables(s).c isa Matrix{Float32}
@inferred logdensityof(ab_model, s)
@inferred logdensityof(ac_model, s)
@inferred logdensityof(bc_model, s)
@inferred logdensityof(abc_model, s)
# Bijectors
abc_bijectors = @inferred bijector(abc_model)
tabc_model = @inferred transformed(abc_model)
ts = @inferred rand(tabc_model)
ms, logjac = @inferred to_model_domain(ts, abc_bijectors)
ℓ = @inferred logdensityof(tabc_model, ts)
@test logdensityof(abc_model, ms) + logjac ≈ ℓ

# RngModel
xoshiro = Xoshiro()
rng_model = RngModel(xoshiro, abc_model)
# Same rng is used both times
Random.seed!(xoshiro, 42)
s1 = rand(Random.default_rng(), rng_model)
Random.seed!(xoshiro, 42)
s2 = rand(xoshiro, rng_model)
@test variables(s1).a == variables(s2).a
@test variables(s1).b == variables(s2).b
@test variables(s1).c == variables(s2).c
@test logdensityof(rng_model, s1) == logdensityof(rng_model, s2)
@test logdensityof(rng_model, s1) == logdensityof(abc_model, s2)
# Bijectors
rng_bijectors = @inferred bijector(rng_model)
rng_tmodel = @inferred transformed(rng_model)
rng_ts = @inferred rand(rng_tmodel)
rng_ms, rng_logjac = @inferred to_model_domain(rng_ts, abc_bijectors)
rng_ℓ = @inferred logdensityof(rng_tmodel, rng_ts)
@test logdensityof(rng_model, rng_ms) + rng_logjac ≈ rng_ℓ

# ComposedModel
c_model = IndependentModel((; c=ProductBroadcastedDistribution(KernelExponential, fill(2.0, 2))))
comp_model = @inferred ComposedModel(ab_model, bc_model, c_model)
s = @inferred rand(comp_model)
@test variables(s).a isa Float16
@test variables(s).b isa Vector{Float32}
@test variables(s).c isa Vector{Float64}
@test logdensityof(comp_model, s) == logdensityof(ab_model, s) + logdensityof(bc_model, s) + logdensityof(c_model, s)
# Bijectors
# Logjac correction must be applied for every prior dist, however multiple priors for same variable do not make sense
comp_model = @inferred ComposedModel(ab_model, c_model)
comp_bijectors = @inferred bijector(comp_model)
@test comp_bijectors.a isa Bijector
@test comp_bijectors.b isa Bijector
@test comp_bijectors.c isa Bijector
comp_tmodel = @inferred transformed(comp_model)
comp_ts = @inferred rand(comp_tmodel)
comp_ms, comp_logjac = @inferred to_model_domain(comp_ts, comp_bijectors)
comp_ℓ = @inferred logdensityof(comp_tmodel, comp_ts)
@test isapprox(logdensityof(comp_model, comp_ms) + comp_logjac, comp_ℓ, rtol=eps(Float16))

# ConditionedModel
a_ind = IndependentModel((; a=a_model))
b_ind = IndependentModel((; b=b_model))
a_sample = rand(a_ind)
b_sample = rand(b_ind)
con_model = ConditionedModel(a_sample, ab_model)

ab_sample = @inferred rand(con_model)
@test variables(ab_sample).a |> typeof == variables(a_sample).a |> typeof
@test variables(ab_sample).b |> typeof == variables(b_sample).b |> typeof
@test variables(ab_sample).a == variables(a_sample).a
@test variables(ab_sample).b != variables(b_sample).b

ℓ = @inferred logdensityof(con_model, b_sample)
@test logdensityof(a_ind, a_sample) + logdensityof(b_ind, b_sample) == ℓ
ℓ = @inferred logdensityof(con_model, ab_sample)
@test logdensityof(a_ind, a_sample) + logdensityof(b_model, variables(ab_sample).b) == ℓ

# Bijectors
con_bijectors = @inferred bijector(con_model)
@test_throws Exception con_bijectors.a
@test con_bijectors.b isa Bijector
con_tmodel = @inferred transformed(con_model)
con_ts = @inferred rand(con_tmodel)
con_ms, con_logjac = @inferred to_model_domain(con_ts, con_bijectors)
con_ℓ = @inferred logdensityof(con_tmodel, con_ts)
@test logdensityof(con_model, con_ms) + con_logjac == con_ℓ
