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

Base.show(io::IO, s::Sample) = print(io, "Sample\n  Log probability: $(log_prob(s))\n  Variable names: $(names(s)) \n  Variable types: $(types(s))")

"""
    names(Sample)
Returns a tuple of the variable names.
"""
names(::Sample{T}) where {T} = T

"""
    types(Sample)
Returns a tuple of the variable types.
"""
types(::Sample{<:Any,V}) where {V} = V

"""
    variables(Sample)
Returns a named tuple of the variables.
"""
variables(s::Sample) = s.variables

"""
    log_prob(sample)
Jacobian-corrected posterior log probability of the sample.
"""
log_prob(sample::Sample) = sample.logp

"""
    +(a, b)
Add the sample `b` to the sample `a`.
The returned sample is of the same type as `a`.
"""
Base.:+(a::Sample, b::Sample) = merge(a, a + variables(b))

"""
    +(a, b)
Add a NamedTuple `b` to the sample `a`.
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
Subtract a NamedTuple `b` from the sample `a`.
"""
function Base.:-(a::Sample, b::NamedTuple)
    sum_nt = map_intersect(.-, variables(a), b)
    @set a.variables = merge(a.variables, sum_nt)
end

"""
    merge(a, b...)
Left-to-Right merges the samples as with bs.
This means the the rightmost variables are kept.
Merging the log probabilities does not make sense without evaluating against the overall model, thus it is -Inf
"""
function Base.merge(a::Sample, b::Sample...)
    merged_variables = merge(variables(a), map(variables, b)...)
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
    # TODO make sure only to use relevant variables, for example only the ones specified by the variable names of the NamedTuple of models.
    # TODO make sure to copy CuArrays to the CPU or we will run out of memory soon
    variables = map(variables, samples)
    TupleVector(variables[start:step:end])
end
