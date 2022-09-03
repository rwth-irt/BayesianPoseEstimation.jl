# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Base: Callable

"""
    FunctionManipulation.jl
The idea is to use generator functions for Measures with a simple way to condition on different parameters.
Partial application of functions can be seen as a way to accomplish this.
Given gen_f(;a,b), partial(gen_f, a=1) returns a function generator of the form gen_b(;b)

Based on ideas from PartialFunctions.jl https://github.com/archermarx/PartialFunctions.jl/blob/master/src/PartialFunctions.jl
Main difference is that mapping the kwargs to args is more flexible and possible in any step of the partial application
"""

# WARN args... and kwargs... lead to Tuple{Symbol} which is non-bits

"""
    partial(f, nt)
Partially applies the named tuple parameters to the function call.
Future parameters will be appended to this function.
"""
partial(f::Callable, nt::NamedTuple) = (x...; y...) -> f(x...; nt..., y...)

"""
    partial(f, t)
Partially applies the tuple parameters to the function call.
Future parameters will be appended to this function.
"""
partial(f::Callable, t::Tuple) = (x...; y...) -> f(t..., x...; y...)

"""
    partial(f, arg)
Partially applies the argument to the function call.
Future parameters will be appended to this function.
"""
partial(f::Callable, arg) = partial(f, (arg,))

# WARN sym::Symbol (; sym => val) not type stable, instead use Val{S}
"""
    partial(f, ::Val{S})
Moves the parameter with Symbol S from the keyword arguments to the positional arguments.
It might behave unexpected in a way, that the parameter is prepend to the positional arguments.
This is necessary to allow that future parameters can be appended to this function.
"""
kwarg_to_arg(f::Callable, ::Val{S}) where {S} = (s, x...; y...) -> f(x...; (; S => s)..., y...)

"""
    ManipulatedFunction
Keep track of the original function and the arguments which are manipulated.
Partial application using anonymous functions leads to ambiguous signatures.
"""
struct ManipulatedFunction{F<:Callable,G<:Callable,T<:Tuple,U<:NamedTuple} <: Function
    func::F
    # Avoid allocations to store the name as string
    original::G
    args::T
    kwargs::U
end

# Enables GPU execution, by bypassing ManipulatedFunction which is not isbits
Broadcast.broadcasted(s::S, mf::ManipulatedFunction, args...) where {S<:Broadcast.BroadcastStyle} = Broadcast.broadcasted(s, mf.func, args...)

function ManipulatedFunction(mf::ManipulatedFunction, nt::NamedTuple)
    func = partial(mf.func, nt)
    ManipulatedFunction(func, mf.original, mf.args, (; mf.kwargs..., nt...))
end

function ManipulatedFunction(mf::ManipulatedFunction, t::Tuple)
    func = partial(mf.func, t)
    ManipulatedFunction(func, mf.original, (mf.args..., t...), mf.kwargs)
end

function ManipulatedFunction(mf::ManipulatedFunction, ::Val{S}) where {S}
    func = kwarg_to_arg(mf.func, Val(S))
    ManipulatedFunction(func, mf.original, (mf.args..., S), mf.kwargs)
end

ManipulatedFunction(mf::ManipulatedFunction, s::Symbol) = ManipulatedFunction(mf, Val(s))
ManipulatedFunction(mf::ManipulatedFunction, x) = ManipulatedFunction(mf, (x,))

fn_name = String ∘ Symbol
ManipulatedFunction(f::Callable) = ManipulatedFunction(f, f, (), (;))
ManipulatedFunction(f::Callable, x) = ManipulatedFunction(ManipulatedFunction(f), x)

"""
    mf(args, kwargs)
Make the function callable, appending `args` and `kwargs`.
Earlier `kwargs` are overridden.
"""
(mf::ManipulatedFunction)(args...; kwargs...) = mf.func(args...; kwargs...)

# Pretty print
Base.show(io::IO, pf::ManipulatedFunction) = print(io, "$(fn_name(pf.original))($(pf.args)...; $(pf.kwargs)...) (partially applied function)")
Base.show(io::IO, ::MIME"text/plain", pf::ManipulatedFunction) = show(io, pf)

"""
    |(f, nt)
Syntactic sugar for ManipulatedFunction(fn, x)
"""
Base.:|(f::Union{Callable,ManipulatedFunction}, x) = ManipulatedFunction(f, x)

"""
    Function(mf)
Convert the ManipulatedFunction to a regular Function
"""
Function(mf::ManipulatedFunction) = mf.func
