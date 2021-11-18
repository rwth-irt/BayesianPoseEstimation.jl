# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 
using MCMCDepth
using Soss, MeasureTheory
using TransformVariables

# Extensions
rand(UniformInterval(1, 2))
logpdf(UniformInterval(0.0, 2.0), 0)
rand(CircularUniform())
logpdf(CircularUniform(), 2π + 0.001)
transform(as(CircularUniform()), 100)

prior_m = @model begin
    r ~ CircularUniform()
    t ~ UniformInterval(1, 3)
    op ~ For(1:3) do _
        Uniform()
    end
end

prior_ℓ(x) = logdensity(prior_m | x)
prior_t = xform(prior_m(;))
θ1 = rand(prior_m)
prior_ℓ(θ1)

obs_m = @model op begin
    o .~ Bernoulli.(op)
end

obs_ℓ(θ, y) = logdensity(obs_m(θ) | y)

y1 = rand(obs_m(θ1))
obs_ℓ(θ1, y1)

post_ℓ = prior_ℓ(θ1) + obs_ℓ(θ1, y1)
inverse(prior_t, θ1)
s1 = Sample(θ1, post_ℓ, prior_t)
state(s1)

# Test limit of transform :)
proposal_m = @model begin
    r ~ Exponential(1)
    t ~ Exponential(1)
    op .~ Exponential.([1, 1, 1])
end
s2 = s1 + rand(proposal_m)
s2 = s2 + rand(proposal_m)
state(s2)
obs_ℓ(state(s2), y1)
