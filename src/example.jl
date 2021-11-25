# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 
using BenchmarkTools
using MCMCDepth
using Soss, MeasureTheory
using TransformVariables

bernoulli_simple = @model o begin
    oc .~ Bernoulli.(o)
end

likelihood_simple = @model oc begin
    y ~ For(oc) do o
        if o
            Normal()
        else
            Exponential()
        end
    end
end

likelihood_mixture = @model o begin
    y ~ For(o) do i
        MixtureMeasure([Normal(), Exponential()], [i, 1.0 - i])
    end
end

likelihood_binary = @model o begin
    y ~ For(o) do i
        BinaryMixture(Normal(), Exponential(), i, 1.0 - i)
    end
end

function posterior_logdensity(model, prior_model, o::T, y::U) where {T,U}
    oc = rand(prior_model(o))
    logdensity(model(oc) | y)
end

function posterior_logdensity_mix(model, o::T, y::U) where {T,U}
    logdensity(model(o) | y)
end

o_ = (; o = rand.(UniformInterval.(zeros(500, 500), 1)))
oc_ = rand(bernoulli_simple(o_))
y_ = rand(likelihood_simple(oc_))
y_ = rand(likelihood_mixture(o_))
y_ = rand(likelihood_binary(o_))
mean(y_.y)
MixtureMeasure([Normal(), Exponential()], [0.1, 1.0 - 0.1])
BinaryMixture(Normal(), Exponential(), 1, 9)
# log_weights_ = (log(w) for w in μ.weights)

@code_warntype posterior_logdensity(likelihood_simple, bernoulli_simple, o_, y_)
@code_warntype posterior_logdensity_mix(likelihood_mixture, o_, y_)
@code_warntype posterior_logdensity_mix(likelihood_binary, o_, y_)
@benchmark posterior_logdensity_mix(likelihood_mixture, o_, y_)
@benchmark posterior_logdensity_mix(likelihood_binary, o_, y_)
@benchmark posterior_logdensity(likelihood_simple, bernoulli_simple, o_, y_)
# TODO logsumexp is way slower than logaddexp
@benchmark logsumexp([1.0, 2.0])
@benchmark logsumexp(1.0, 2.0)
@benchmark logaddexp(1.0, 2.0)

# TODO not type safe: For(eachindex(o))
# TODO not type safe: MixtureMeasure([Normal(), Exponential()], [i, 1 - i] (1-i might be Int)
test_simple = @model o begin
    y ~ For(o) do i
        MixtureMeasure([Normal(), Exponential()], [i, 1.0 - i])
    end
end

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
