# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# bundle_samples for the Sample type
using AbstractMCMC, TupleVectors
using Accessors
using TransformVariables

"""
    Sample(vars, logp)
Consists of the unconstrained state `vars` and the corrected posterior probability `logp=logpₓ(t(θ)|z)+logpₓ(θ)+logjacdet(t(θ))`.
Samples are generically typed by `T` for the variable names and `U` to specify their respective domain transformation.
"""
struct Sample{T,V<:Tuple{Vararg{AbstractVariable}}}
    vars::NamedTuple{T,V}
    logp::Float64
end

Base.show(io::IO, s::Sample) = print(io, "Sample\n  Log probability: $(logp(s))\n  Variable names: $(names(s))")

"""
    names(Sample)
Returns a tuple of the variable names.
"""
names(::Sample{T}) where {T} = T

"""
    vars(Sample)
Returns a named tuple of the vars.
"""
vars(s::Sample) = s.vars

"""
    state(s, var_name)
Returns the state in the model domain of the variable `var_name`.
"""
model_value(s::Sample, var_name::Symbol) = model_value(vars(s)[var_name])

"""
    raw_state(s, var_name)
Returns the state in the unconstrained domain of the variable `var_name`.
"""
raw_value(s::Sample, var_name::Symbol) = raw_value(vars(s)[var_name])

"""
    logp(s)
Jacobian-corrected posterior log probability of the sample.
"""
logp(s::Sample) = s.logp

"""
    flatten(x)
Flattens x to return a 1D array.
"""
flatten(x) = collect(Iterators.flatten(x))

"""
    add!(a, b)
Add a raw state `θ` to the raw state (unconstrained domain) of the sample `s`.
Modifies a
"""
function add!(a::Sample, b::Sample)
    a_ranges = ranges(a)
    b_ranges = ranges(b)
    for var_name in keys(a_ranges)
        if var_name in keys(b_ranges)
            a.θ[a_ranges[var_name]] = a.θ[a_ranges[var_name]] + b.θ[b_ranges[var_name]]
        end
    end
    a
end

"""
    add!(a, b)
Add a raw state `θ` to the raw state (unconstrained domain) of the sample `s`.
Optimized case for two samples of the same type.
Modifies a
"""
function add!(a::Sample{T}, b::Sample{T}) where {T}
    for (i, v) in enumerate(b.θ)
        a.θ[i] = a.θ[i] + v
    end
end

"""
    map_intersect(f, a, b)
Maps the function `f` over the intersection of the keys of `a` and `b`.
Returns a NamedTuple with the same keys as `a` which makes it type-stable.
"""
function map_intersect(f, a::NamedTuple{T}, b::NamedTuple) where {T}
    vars = map(keys(a)) do k
        if k in keys(b)
            f(a[k], b[k])
        else
            a[k]
        end
    end
    NamedTuple{T}(vars)
end

# TODO this implies, that all necessary variables are expected to be present in the sample. Thus, proposals need to include internal variables like the expected depth. Filter out irrelevant variables when returning the state in the sampler. Alternatively only calculate internal variables in the likelihood function.
"""
    map_models(f, models, vars; default)
Map the function `f(model, variable, variables)` over each `model` and `variable`.
All other required `variables` are passed into `f` as context.
If no intersection exists, the `default` value is used.
For log-densities the reasonable default is 0.0 (summation).
Non-logarithmic densities should use 1.0 as default (product).
"""
map_models(f, models::NamedTuple, variables::NamedTuple; default=0.0) =
    map(keys(models)) do k
        if k in keys(variables)
            f(models[k], variables[k], variables)
        else
            default
        end
    end

"""
    map_models(f, models, sample; default)
Map the function `f(model, variable, variables)` over each `model` and variable in `sample`.
All other required `variables` are passed into `f` as context.
If no intersection exists, the `default` value is used.
For log-densities the reasonable default is 0.0 (summation).
Non-logarithmic densities should use 1.0 as default (product).
"""
map_models(f, models::NamedTuple, sample::Sample; default=0.0) = map_models(f, models, vars(sample), default)

"""
    +(a, b)
Add the sample `b` to the sample `a`.
The returned sample is of the same type as `a`.
"""
function Base.:+(a::Sample, b::Sample)
    sum_nt = map_intersect(+, vars(a), vars(b))
    @set a.vars = sum_nt
end

"""
    +(a, b)
Add a NamedTuple `b` to the sample `a`.
"""
function Base.:+(a::Sample, b::NamedTuple)
    sum_nt = map_intersect(+, vars(a), b)
    @set a.vars = sum_nt
end

"""
    -(a, b)
Subtract the raw states (unconstrained domain) of two samples.
Only same type is supported to prevent surprises in the return type.
"""
function Base.:-(a::Sample, b::Sample)
    sum_nt = map_intersect(-, vars(a), vars(b))
    @set a.vars = sum_nt
end

"""
    -(a, b)
Subtract a NamedTuple `b` from the sample `a`.
"""
function Base.:-(a::Sample, b::NamedTuple)
    sum_nt = map_intersect(-, vars(a), b)
    @set a.vars = sum_nt
end

"""
    merge(a, b...)
Left-to-Right merges the samples as with bs.
This means the the rightmost variables are kept.
Merging the log probabilities does not make sense without evaluating against the overall model, thus it is -Inf
"""
function Base.merge(a::Sample, b::Sample...)
    merged_vars = merge(vars(a), map(vars, b)...)
    Sample(merged_vars, -Inf)
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
    vars = map(vars, samples)
    TupleVector(vars[start:step:end])
end
