# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Base: Callable

"""
    FunctionManipulation.jl
The idea is to use generator functions for Distributions with a simple way to condition on different parameters.
Partial application of functions can be seen as a way to accomplish this.
Given gen_f(;a,b), gen_f | (; a=1) returns a function generator of the form gen_b(;b)

Based on ideas from PartialFunctions.jl https://github.com/archermarx/PartialFunctions.jl/blob/master/src/PartialFunctions.jl

# TODO this has proven difficult using chained anonymous functions
Main difference is that mapping the kwargs to args is more flexible and possible in any step of the partial application
"""

"""
    ManipulatedFunction
Keep track of the original function and the arguments which are manipulated.
Partial application using anonymous functions leads to ambiguous signatures.
"""
struct ManipulatedFunction{names,F<:Function,G<:Callable,T<:Tuple,U<:NamedTuple} <: Function
    # Parametric names required for type stable for moving kwargs to args
    # Use anonymous function for Callable, since type constructors are UnionAll / Union{Function,Type}
    func::F
    # Avoid allocations to store the name as string
    original::G
    args::T
    kwargs::U
end

# simplifies parametric constructor
ManipulatedFunction(func::F, original::G, args::T, kwargs::U, names=()) where {F,G,T,U} = ManipulatedFunction{names,F,G,T,U}(func, original, args, kwargs)

ManipulatedFunction(f::Function) = ManipulatedFunction(f, f, (), (;))
# Callable support. Since constructor can be function or type, convert to anonymous function.
ManipulatedFunction(::Type{T}) where {T} = ManipulatedFunction((x...; y...) -> T(x..., y...), T, (), (;))
ManipulatedFunction(f::Callable, x) = ManipulatedFunction(ManipulatedFunction(f), x)

# partial applications of an existing ManipulatedFunction
ManipulatedFunction(mf::ManipulatedFunction{names}, t::Tuple) where {names} = ManipulatedFunction(mf.func, mf.original, (mf.args..., t...), mf.kwargs, names)

ManipulatedFunction(mf::ManipulatedFunction{names}, nt::NamedTuple) where {names} = ManipulatedFunction(mf.func, mf.original, mf.args, (; mf.kwargs..., nt...), names)

ManipulatedFunction(mf::ManipulatedFunction{names}, t::Tuple{Vararg{<:Symbol}}) where {names} = ManipulatedFunction(mf.func, mf.original, mf.args, mf.kwargs, (names..., t...))

ManipulatedFunction(mf::ManipulatedFunction, x) = ManipulatedFunction(mf, (x,))

"""
    |(f, nt)
Syntactic sugar for ManipulatedFunction(fn, x)
"""
Base.:|(f::Union{Callable,ManipulatedFunction}, x) = ManipulatedFunction(f, x)


# ManipulatedFunction is callable
function (mf::ManipulatedFunction{names})(args...; kwargs...) where {names}
    # these args are mapped to kwargs using the parametric names
    kw2arg = args[end-length(names)+1:end]
    nt = NamedTuple{names}(kw2arg)
    # these args are not mapped to kwargs
    remaining = args[begin:end-length(names)]
    mf.original(mf.args..., remaining...; mf.kwargs..., nt..., kwargs...)
end

# Pretty print
function Base.show(io::IO, mf::ManipulatedFunction{names}) where {names}
    if mf.kwargs == (;)
        print(io, "$(fn_name(mf.original))$((mf.args..., names...))")
    else
        print(io, "$(fn_name(mf.original))$((mf.args..., names...)); $(mf.kwargs))")
    end
end

Base.show(io::IO, ::MIME"text/plain", mf::ManipulatedFunction) = show(io, mf)
fn_name = String ∘ Symbol
