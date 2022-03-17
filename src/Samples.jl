# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# bundle_samples for the Sample type
using AbstractMCMC, TupleVectors
using Accessors
using MeasureTheory, Soss
using Random
using TransformVariables

"""
    Sample(θ, p, t)
Might have a constrained parameter Domain, e.g. θᵢ ∈ ℝ⁺.
Consists of the unconstrained raw state `θ::Vector{Float64}`, the corrected posterior probability `logp=logpₓ(t(θ)|z)+logpₓ(θ)+logjacdet(t(θ))` and a transformation rule `t`.
Samples are generically typed by `T` for the variable names and `U` to specify their respective domain transformation.
"""
struct Sample{T,U<:TransformVariables.TransformTuple}
    θ::Vector{Float64}
    logp::Float64
    t::U
    # Extract variable names from the TransformTuple
    Sample(θ::AbstractVector, p, t::TransformVariables.TransformTuple{<:NamedTuple{T}}) where {T} = new{T,typeof(t)}(θ, p, t)
end

Base.show(io::IO, s::Sample) = print(io, "Sample\n  Log probability: $(logp(s))\n  Variables: $(variables(s))")


"""
    Sample(θ, p, t)
Create a sample from `θ::NamedTuple` containing the current state and its probability `p`.
`t` is a transformation from the unconstrained to the constrained space.
"""
function Sample(θ::NamedTuple, p, t::TransformVariables.TransformTuple)
    # Memory access error using Revise?
    θ_raw = inverse(t, θ)
    Sample(θ_raw, p, t)
end

"""
    Sample(m)
Create a sample by sampling from a model.
"""
function Sample(rng::AbstractRNG, m::Soss.AbstractModel)
    θ = rand(rng, m)
    # xform requires ConditionalModel, conditioning on nothing does not change the model except converting it a ConditionalModel
    tr = xform(m | (;))
    Sample(θ, -Inf, tr)
end

"""
    Sample(m)
Create a sample by sampling from a model.
"""
Sample(m::Soss.AbstractModel) = Sample(Random.GLOBAL_RNG, m)

"""
    variables(Sample)
Returns a tuple of the variable names.
"""
variables(::Sample{T}) where {T} = T

"""
    state(s)
State of the parameters in the model domain.
Returns a `NamedTuple` with the variable names and their model domain values.
"""
state(s::Sample) = transform(s.t, s.θ)

"""
    raw_state(s)
State of the parameters in the unconstrained domain.
Returns a `NamedTuple` with the variable names and their model domain values.
"""
raw_state(s::Sample{T}) where {T} = NamedTuple{T}(s.θ)

"""
    logp(s)
Jacobian-corrected posterior log probability of the sample.
"""
logp(s::Sample) = s.logp

"""
    logdensity(m, s)
Non-corrected logdensity of the of the sample `s` given the measure `m`.
"""
MeasureTheory.logdensity(m::Soss.AbstractMeasure, s::Sample) = logdensity(m, state(s))

"""
    logdensity(m, s)
Non-corrected logdensity of the of the sample `s` given the model `m`.
"""
# Required to solve ambiguity for ConditionalModel
MeasureTheory.logdensity(m::Soss.ConditionalModel, s::Sample) = logdensity(m, state(s))

"""
    transform_logdensity(m, s)
Jacobian-corrected logdensity of the of the sample `s` given the model `m`.
"""
function TransformVariables.transform_logdensity(m::AbstractMeasure, s::Sample)
    f(y) = logdensity(m, y)
    transform_logdensity(s.t, f, s.θ)
end

"""
    transform_logdensity(m, s)
Jacobian-corrected logdensity of the of the sample `s` given the model `m`.
"""
function TransformVariables.transform_logdensity(m::Soss.ConditionalModel, s::Sample)
    f(y) = logdensity(m, y)
    transform_logdensity(s.t, f, s.θ)
end

"""
    flatten(x)
Flattens x to return a 1D array.
"""
flatten(x) = collect(Iterators.flatten(x))

"""
    ranges(nt)
Helps to find which range of the raw state belongs to which variable.
Returns a dictionary of UnitRange for each key in the NamedTuple
"""
function ranges(nt::NamedTuple{T}) where {T}
    d = Dict{Symbol,UnitRange{Int64}}()
    iter = 1
    for var_name in T
        l = length(nt[var_name])
        d[var_name] = iter:(iter+l-1)
        iter = iter + l
    end
    return d
end

"""
    ranges(nt)
Helps to find which range of the raw state belongs to which variable.
Returns a dictionary of UnitRange for each key in the NamedTuple
"""
ranges(s::Sample) = ranges(state(s))

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
    +(a, b)
Add a raw state `θ` to the raw state (unconstrained domain) of the sample `s`.
"""
Base.:+(a::Sample, b::Sample) = add!(deepcopy(a), b)

"""
    +(a, b)
Add the raw states (unconstrained domain) of two samples.
Optimized case for two samples of the same type.
"""
Base.:+(a::Sample{T}, b::Sample{T}) where {T} = @set a.θ = a.θ + b.θ

"""
    +(a, b)
Add a NamedTuple `b` interpreted as raw state to the raw state of two `a``.
Modifies `a`.
"""
function add!(a::Sample, b::NamedTuple)
    a_ranges = ranges(a)
    b_ranges = ranges(b)
    b_values = flatten(b)
    for var_name in keys(a_ranges)
        if var_name in keys(b)
            a.θ[a_ranges[var_name]] = a.θ[a_ranges[var_name]] + b_values[b_ranges[var_name]]
        end
    end
    a
end

"""
    +(a, b)
Add a NamedTuple `b` interpreted as raw state to the raw state of two `a``.
"""
Base.:+(a::Sample, b::NamedTuple) = add!(deepcopy(a), b)

"""
    -(a, b)
Subtract the raw states (unconstrained domain) of two samples.
Only same type is supported to prevent surprises in the return type.
"""
function Base.:-(a::Sample, b::Sample)
    negative_b = @set b.θ = -b.θ
    a + negative_b
end


"""
    merge(a, b...)
Left-to-Right merges the samples as with bs.
This means the the rightmost state is used.
The original transformation rule of as is kept to ensure compatibility with the original sample.
Merging the log probabilities does not make sense without evaluating against the overall model, thus it is -Inf
"""
function Base.merge(a::Sample, b::Sample...)
    merged_state = merge(state(a), map(x -> state(x), b)...)
    Sample(merged_state, -Inf, a.t)
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
    start = 1,
    step = 1
)
    states = map(state, samples)
    TupleVector(states[start:step:end])
end
