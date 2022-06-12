# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# bundle_samples for the Sample type
using AbstractMCMC, TupleVectors
using Accessors

"""
    Sample(variables, logp)
Consists of the unconstrained state `variables` and the corrected posterior probability `logp=logpₓ(t(θ)|z)+logpₓ(θ)+logjacdet(t(θ))`.
Samples are typed by `T,V` as the internal named tuple for the variable names types.
"""
struct Sample{T,V<:Tuple{Vararg{AbstractVariable}}}
    variables::NamedTuple{T,V}
    logp::Float64
end

Base.show(io::IO, s::Sample) = print(io, "Sample\n  Log probability: $(logp(s))\n  Variable names: $(names(s))")

"""
    names(Sample)
Returns a tuple of the variable names.
"""
names(::Sample{T}) where {T} = T

"""
    variables(Sample)
Returns a named tuple of the variables.
"""
variables(s::Sample) = s.variables

"""
    state(sample, var_name)
Returns the state in the model domain of the variable `var_name`.
"""
model_value(sample::Sample, var_name::Symbol) = model_value(variables(sample)[var_name])

"""
    raw_state(sample, var_name)
Returns the state in the unconstrained domain of the variable `var_name`.
"""
raw_value(sample::Sample, var_name::Symbol) = raw_value(variables(sample)[var_name])

"""
    to_model_variables(sample)
Converts all the variables of the `sample` to `ModelVariable`.
"""
to_model_variables(sample::Sample) = @set sample.variables = map(ModelVariable, variables(sample))

"""
    to_sample_variables(sample)
Converts all the variables of the `sample` to `SampleVariable`.
"""
to_sample_variables(sample::Sample) = @set sample.variables = map(SampleVariable, variables(sample))

"""
    logp(sample)
Jacobian-corrected posterior log probability of the sample.
"""
logp(sample::Sample) = sample.logp

"""
    flatten(x)
Flattens x to return a 1D array.
"""
flatten(x) = collect(Iterators.flatten(x))

"""
    map_intersect(f, a, b, default)
Maps the function `f` over the intersection of the keys of `a` and `b`.
Uses the value of `default`, which may be a function of `value(a[i])`, if no matching key is found in `b`.
Returns a NamedTuple with the same keys as `a` which makes it type-stable.
"""
map_intersect(f, a::NamedTuple{A}, b::NamedTuple, default) where {A} = NamedTuple{A}(map_intersect_(f, a, b, default))

# Barrier for type stability of getindex?
map_intersect_(f, a::NamedTuple{A}, b::NamedTuple{B}, default) where {A,B} =
    map(A) do k
        if k in B
            f(a[k], b[k])
        else
            default
        end
    end

map_intersect_(f, a::NamedTuple{A}, b::NamedTuple{B}, default_fn::Function) where {A,B} =
    map(A) do k
        if k in B
            f(a[k], b[k])
        else
            default_fn(value([k]))
        end
    end

"""
    map_intersect(f, a, b)
Maps the function `f` over the intersection of the keys of `a` and `b`.
Uses the value of `a` if no matching key is found in `b`.
Returns a NamedTuple with the same keys as `a` which makes it type-stable.
"""
function map_intersect(f, a::NamedTuple{A}, b::NamedTuple{B}) where {A,B}
    # Type stability is delicate
    filtered_keys = filter(in(A), B)
    filtered_values = map(f, a[filtered_keys], b[filtered_keys])
    NamedTuple{filtered_keys}(filtered_values)
end

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
    sum_nt = map_intersect(+, variables(a), b)
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
    sum_nt = map_intersect(-, variables(a), b)
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
