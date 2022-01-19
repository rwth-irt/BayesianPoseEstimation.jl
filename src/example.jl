# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 
using BenchmarkTools
using Soss, MeasureTheory
using MCMCDepth
using TransformVariables

# Samples
using Soss, MeasureTheory
using MCMCDepth
sample_m = @model begin
    b ~ Normal(0, 2)
    a ~ Normal(b, 1)
end
s1 = Sample(sample_m)

sample_m2 = @model begin
    #a ~ Normal(0, 1)
    b ~ Normal(1, 2)
    c .~ Normal.([0, 0], 1)
end
s2 = Sample(sample_m2)
s1 + s2
s1.θ + reverse(s2.θ)
s1 - s2
s1.θ - reverse(s2.θ)
nt = state(s2)
MCMCDepth.flatten(nt)
s1 + nt

m1 = @model begin
    o .~ UniformInterval.([0 0; 0 0], 1)
    b ~ Uniform()
end
s1 = Sample(m1)

m2 = @model begin
    o .~ UniformInterval.([0 0; 0 0], 1)
end
s2 = Sample(m2)
merge(s1, s2)

# Proposals
prio_m = @model begin
    a ~ Uniform()
end
prop_m = @model begin
    a ~ Normal(1, 0.1)
end
ind_prop = IndependentProposal(prio_m)
s = propose(ind_prop)
sym_prop = SymmetricProposal(prop_m)
s2 = propose(sym_prop, s)
prop = Proposal(prop_m)
s3 = propose(sym_prop, s2)
transition_probability(prop, s3, s2)
transition_probability(prop, s2, s3)
transition_probability(sym_prop, s2, s3)

# Performance
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
transform(as((; a = as(Array, as○, 2), b = as_unit_interval)), [100; 200; 10])

circ_m = @model begin
    a ~ For(1:3) do i
        UniformInterval(1, 2)
    end
    b ~ Normal()
end
sample = Sample(circ_m)
state(sample)
log_probability(sample)

prior_m = @model begin
    r ~ CircularUniform()
    t ~ UniformInterval(1, 3)
    op ~ For(1:3) do _
        Uniform()
    end
end

prior_ℓ(x) = logdensity(prior_m | x)
prior_t = xform(prior_m | (;))
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

# Gibbs
using MCMCDepth
using Soss, MeasureTheory
m = @model begin
    a ~ Normal(0, 1)
    b ~ Normal(a, 1)
end

ip = IndependentProposal(m)
s1 = propose(ip)
transition_probability(ip, s1, s1)

#b_m = Soss.likelihood(m, :b)
#s2 = Sample(b_m)
gpb = GibbsProposal{SymmetricProposal}(m, :b)
s2 = propose(gpb, s1)
gpa = GibbsProposal{SymmetricProposal}(m, :a)
s3 = propose(gpa, s2)

transition_probability(gp, s2, s1)
logdensity(b_m(state(s2)), s1)

gp2 = GibbsProposal(gp)
s3 = propose(gp2, s2)
transition_probability(gp2, s3, s2)

mh = MetropolisHastings(ip)
Gibbs(mh, MetropolisHastings(gp))

# TransformVariables
using TransformVariables
t = as_unit_interval
transform(t, -200)
inverse(t, eps())