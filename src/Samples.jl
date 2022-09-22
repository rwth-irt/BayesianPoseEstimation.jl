# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# bundle_samples for the Sample type
using AbstractMCMC, TupleVectors
using Accessors

"""
    Sample{T,V}(variables, logp)
Consists of the state `variables` and the corrected posterior probability `logp=logpₓ(t(θ)|z)+logpₓ(θ)+logjacdet(t(θ))`.
Samples are typed by `T,V` as the internal named tuple for the variable names types.
"""
struct Sample{T,V}
    variables::NamedTuple{T,V}
    logp::Float64
end

Base.show(io::IO, s::Sample) = print(io, "Sample\n  Log probability: $(logprob(s))\n  Variable names: $(names(s)) \n  Variable types: $(types(s))")

"""
    names(sample)
Returns a tuple of the variable names.
"""
names(::Sample{T}) where {T} = T

"""
    types(sample)
Returns a tuple of the variable types.
"""
types(::Sample{<:Any,V}) where {V} = V

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
function to_model_domain(s::Sample{T}, bijectors::NamedTuple{<:Any,<:Tuple{Vararg{Bijector}}}) where {T}
    with_logjac = map_intersect((b, v) -> with_logabsdet_jacobian(b, v), map(inverse, bijectors), variables(s))
    tr_vars = map(first, with_logjac)
    model_sample = @set s.variables = merge(s.variables, tr_vars)
    logjac = reduce(.+, values(map(last, with_logjac)); init=0)
    model_sample, logjac
end

"""
    to_unconstrained_domain(sample, bijector)
Transform the sample to ℝⁿ by transforming the (some) variables of the sample using the bijectors.
"""
function to_unconstrained_domain(sample::Sample, bijectors::NamedTuple)
    tr_variables = merge(variables(sample), map_intersect((b, v) -> b(v), bijectors, variables(sample)))
    Sample(tr_variables, logprob(sample))
end

"""
    logprob(sample)
(Logjac corrected) posterior log probability of the sample.
"""
logprob(sample::Sample) = sample.logp

"""
    +(a, b)
Add the raw states (unconstrained domain) of two samples.
The returned sample is of the same type as `a`.
"""
Base.:+(a::Sample, b::Sample) = merge(a, a + variables(b))

"""
    +(a, b)
Add a NamedTuple `b` to the raw values of sample `a`.
"""
function Base.:+(a::Sample, b::NamedTuple)
    sum_nt = map_intersect(.+, variables(a), b)
    @set a.variables = merge(a.variables, sum_nt)
end

"""
    -(a, b)
Subtract the raw states (unconstrained domain) of two samples.
Only same type is supported to prevent surprises in the return type.
"""
Base.:-(a::Sample, b::Sample) = merge(a, a - variables(b))

"""
    -(a, b)
Subtract a NamedTuple `b` from the raw values of sample `a`.
"""
function Base.:-(a::Sample, b::NamedTuple)
    sum_nt = map_intersect(.-, variables(a), b)
    @set a.variables = merge(a.variables, sum_nt)
end

"""
    merge(a, b...)
Left-to-Right merges the samples.
This means the the rightmost variables are kept.
Merging the log probabilities does not make sense without evaluating against the overall model, thus it is -Inf
"""
Base.merge(a::Sample, b::Sample...) = merge(a, variables.(b)...)

function Base.merge(a::Sample, b::NamedTuple...)
    merged_variables = merge(variables(a), b...)
    Sample(merged_variables, -Inf)
end

"""
    bundle_samples(samples, model, sampler, state, chain_type[; kwargs...])
Bundle all `samples` that were sampled from the `model` with the given `sampler` in a chain.
The final `state` of the `sampler` can be included in the chain. The type of the chain can
be specified with the `chain_type` argument.
By default, this method returns `samples`.
"""
function AbstractMCMC.bundle_samples(
    samples::Vector{<:Sample},
    ::AbstractMCMC.AbstractModel,
    ::AbstractMCMC.AbstractSampler,
    ::Any,
    ::Type{TupleVector};
    start=1,
    step=1
)
    # TODO make sure only to use relevant variables, for example only the ones specified by the variable names of the NamedTuple of the prior.
    variables = variables.(samples)
    TupleVector(variables[start:step:end])
end
