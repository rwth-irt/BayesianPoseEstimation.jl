# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# TODO should I enforce log_prob and log_like to be stored on CPU RAM? Avoids some scalar indexing and GPU out-of-memory errors. Might hinder massively parallelized calculations.
"""
    Sample{T,V}(variables, logp, logℓ)
Consists of the state `variables θ`, the log likelihood `log_like(t(θ)|z)+`, and the corrected log posterior probability `log_prob=log_liket(θ)|z)+logp(θ)+logjacdet(t(θ))`.
Samples are typed by `T,V` as the internal named tuple for the variable names types.
"""
struct Sample{T<:NamedTuple,L<:Union{Real,Array},P<:Union{Real,Array}}
    variables::T
    log_prob::L
    log_like::P
end

"""
    Sample(variables)
Generate a new sample from a named tuple of variables.
By default -Inf is assigned as log probability.
"""
Sample(variables::NamedTuple) = Sample(variables, -Inf, -Inf)

Base.show(io::IO, s::Sample) = print(io, "Sample\n  log probability: $(logprob(s))\n log likelihood: $(loglike(s))\n Variable names: $(names(s)) \n  Variable types: $(types(s))")

"""
    set_logprob(sample, log_prob)
Immutable update the log probability of the sample.
The original is untouched and a new sample returned. 
"""
set_logprob(sample::Sample, log_prob) = @set sample.log_prob = log_prob

"""
    set_loglike(sample, log_like)
Immutable update the log likelihood of the sample.
The original is untouched and a new sample returned. 
"""
set_loglike(sample::Sample, log_like) = @set sample.log_like = log_like

"""
    names(sample)
Returns a tuple of the variable names.
"""
names(::Sample{<:NamedTuple{T}}) where {T} = T

"""
    types(sample)
Returns a tuple of the variable types.
"""
types(::Sample{<:NamedTuple{<:Any,T}}) where {T} = T

"""
    variables(sample)
Returns a named tuple of the raw variables ∈ ℝⁿ.
"""
variables(s::Sample) = s.variables

"""
    to_model_domain(sample, bijectors)
Transforms the sample to the model domain by using the inverse transform of the provided bijectors.
The logjac correction is calculated in the same kernel.
Returns (variables, logabsdetjac)
"""
function to_model_domain(s::Sample, bijectors::NamedTuple)
    with_logjac = map_intersect((b, v) -> with_logabsdet_jacobian(inverse(b), v), bijectors, variables(s))
    tr_vars = map(first, with_logjac)
    model_sample = @set s.variables = merge(s.variables, tr_vars)
    logjac = reduce(add_logdensity, values(map(last, with_logjac)); init=0)
    model_sample, logjac
end

"""
    to_unconstrained_domain(sample, bijector)
Transform the sample to ℝⁿ by transforming the affected variables of the sample using the bijectors.
"""
function to_unconstrained_domain(sample::Sample, bijectors::NamedTuple)
    tr_variables = merge(variables(sample), map_intersect((b, v) -> b(v), bijectors, variables(sample)))
    Sample(tr_variables, logprob(sample), loglike(sample))
end

"""
    logprob(sample)
(Logjac corrected) posterior log probability of the sample.
"""
logprob(sample::Sample) = sample.log_prob

"""
    loglike(sample)
(Logjac corrected) posterior log likelihood of the sample.
"""
loglike(sample::Sample) = sample.log_like

"""
    getindex(sample, idx)
Returns a sample with only a subset of the variables.
"""
Base.getindex(sample::Sample, ::Val{T}) where {T} = Sample(sample.variables[T], -Inf, -Inf)

"""
    merge(a, b)
Left-to-Right merges the samples.
This means the the rightmost variables are kept.
Merging the log probabilities does not make sense without evaluating against the overall model, thus it is -Inf
"""
function Base.merge(a::Sample, b::NamedTuple)
    vars = merge(variables(a), b)
    Sample(vars)
end
Base.merge(a::Sample, b::Sample) = merge(a, variables(b))

"""
    merge(a, b)
Left-to-Right merges the samples by mapping f(variables(a), variables(b)).
This means the the rightmost variables are kept.
Merging the log probabilities does not make sense without evaluating against the overall model, thus it is -Inf
"""
function map_merge(f, a::Sample, b::NamedTuple)
    vars = map_intersect(f, variables(a), b)
    Sample(merge(a.variables, vars))
end
map_merge(f, a::Sample, b::Sample) = map_merge(f, a, variables(b))

"""
    ⊕(a, b)
Add the raw states (unconstrained domain) of two samples.
The returned sample is of the same type as `a`.
Uses additive operator ⊕ which supports the quaternion tangent space in KernelDistributions.jl.
"""
KernelDistributions.:⊕(a::Sample, b::NamedTuple) = map_merge(.⊕, a, b)
KernelDistributions.:⊕(a::Sample, b::Sample) = a ⊕ variables(b)

"""
    ⊖(a, b)
Subtract the raw states (unconstrained domain) of two samples.
Only same type is supported to prevent surprises in the return type.
Uses subtractive operator ⊖ which supports the quaternion tangent space in KernelDistributions.jl.
"""
KernelDistributions.:⊖(a::Sample, b::NamedTuple) = map_merge(.⊖, a, b)
KernelDistributions.:⊖(a::Sample, b::Sample) = a ⊖ variables(b)

