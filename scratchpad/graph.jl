using Bijectors
using BenchmarkTools
using DensityInterface
using MCMCDepth
using Random
using Test
using Unrolled

abstract type AbstractNode{name,childnames} end
Base.Broadcast.broadcastable(x::AbstractNode) = Ref(x)

# realize the distribution
(node::AbstractNode)(x...) = model(node).(x...)
(node::AbstractNode)(nt::NamedTuple) = node(argvalues(node, nt)...)

# Override if required for specific implementations
argvalues(::AbstractNode{<:Any,childnames}, nt) where {childnames} = values(nt[childnames])
varvalue(::AbstractNode{varname}, nt) where {varname} = nt[varname]

nodes(node::AbstractNode) = node.nodes
model(node::AbstractNode) = node.model

# typestable rand & logdensityof conditioned on variables - do not traverse
# directly support fn(node, nt, args...) of traverse
# dispatching on the node type makes it extendable / hackable
merge_value(variables, ::AbstractNode{name}, value) where {name} = (; variables..., name => value)

rand_barrier(node::AbstractNode{<:Any,()}, variables::NamedTuple, rng::AbstractRNG, dims...) = rand(rng, node(variables), dims...)
# dims... only for leafs, otherwise the dimensioms potentiate
rand_barrier(node::AbstractNode{<:Any}, variables::NamedTuple, rng::AbstractRNG, dims...) = rand(rng, node(variables))

logdensityof_barrier(node::AbstractNode, variables::NamedTuple) = logdensityof(node(variables), varvalue(node, variables))

bijector_barrier(node::AbstractNode, variables::NamedTuple) = bijector(node(variables))

# fn first to enable do syntax
function traverse(fn, node::AbstractNode{varname}, variables::NamedTuple{names}, args...) where {varname,names}
    # Termination: Value already available (conditioned on or calculate via another path)
    if varname in names
        return variables
    end
    # Conditional = values from other nodes required, compute depth first
    for n in nodes(node)
        variables = traverse(fn, n, variables, args...)
    end
    # Finally the internal dist can be realized and the value for this node can be merged
    value = fn(node, variables, args...)
    merge_value(variables, node, value)
end

Base.rand(rng::AbstractRNG, node::AbstractNode{varname}, dims::Integer...) where {varname} = traverse(rand_barrier, node, (;), rng, dims...)

DensityInterface.logdensityof(node::AbstractNode, nt::NamedTuple) =
    reduce(.+, traverse(node, (;)) do n, _
        logdensityof_barrier(n, nt)
    end)

function Bijectors.bijector(node::AbstractNode)
    variables = rand(node)
    traverse(node, (;), variables) do n, _...
        bijector_barrier(n, variables)
    end
end

##################

struct VariableNode{varname,names,M<:Function,N<:NamedTuple{names}} <: AbstractNode{varname,names}
    # Must be function to avoid UnionAll type instabilities
    model::M
    nodes::N
end

VariableNode(varname::Symbol, model::M, nodes::N) where {names,M<:Function,N<:NamedTuple{names}} = VariableNode{varname,names,M,N}(model, nodes)

function VariableNode(varname::Symbol, ::Type{M}, nodes::N) where {names,M,N<:NamedTuple{names}}
    # Workaround so D is not UnionAll but interpreted as constructor
    wrapped = (x...) -> M(x...)
    VariableNode{varname,names,typeof(wrapped),N}(wrapped, nodes)
end

##################

# TODO Idea is that this node just wraps another one and thus shares the same varname and argnames
# TODO makes it harder to compile into static graph? - not really, simply merge from the right as (;varname=this_node)
struct ModifierNode{varname,argnames,M,N<:AbstractNode{varname,argnames}} <: AbstractNode{varname,argnames}
    model::M
    wrapped::N
end

function rand_barrier(node::ModifierNode, variables::NamedTuple, rng::AbstractRNG, dims...)
    wrapped_value = rand_barrier(node.wrapped, variables, rng, dims...)
    rand(rng, node(variables), wrapped_value)
end

function logdensityof_barrier(node::ModifierNode, variables::NamedTuple)
    ℓ = logdensityof_barrier(node.wrapped, variables)
    logdensityof(node(variables), varvalue(node, variables), ℓ)
end

bijector_barrier(node::ModifierNode, variables::NamedTuple) = bijector_barrier(node.wrapped, variables)

# TODO Pass the node into the model to evaluate the internal logdensity and be able to modify it? Is there a cleaner way?
# argvalues(node::ModifierNode{varname,childnames}, nt) where {varname,childnames} = (node.wrapped, values(nt[(varname, childnames...)])...)
# varvalue(::ModifierNode{varname}, nt) where {varname} = nt[varname]
nodes(node::ModifierNode) = (node.wrapped,)

struct SimpleModifierModel end

# Construct with same args as wrapped model
SimpleModifierModel(args...) = SimpleModifierModel()

Base.rand(::AbstractRNG, model::SimpleModifierModel, value) = 10 .* value

DensityInterface.logdensityof(model::SimpleModifierModel, x, ℓ) = ℓ + one(ℓ)

function Bijectors.bijector(node::AbstractNode)
    sample = rand(node)
    traverse(node, (;), sample) do n, _...
        bijector_barrier(n, sample)
    end
end

##################

a = VariableNode(:a, KernelUniform, (;))
b = VariableNode(:b, KernelExponential, (;))
c = VariableNode(:c, KernelNormal, (; a=a, b=b))
d = VariableNode(:d, KernelNormal, (; c=c, b=b))
d_mod = ModifierNode(SimpleModifierModel, d)

nt = rand(Random.default_rng(), d)
ℓ = logdensityof(d, nt)
@test ℓ == logdensityof(KernelUniform(), nt.a) + logdensityof(KernelExponential(), nt.b) + logdensityof(KernelNormal(nt.a, nt.b), nt.c) + logdensityof(KernelNormal(nt.c, nt.b), nt.d)
bij = bijector(d)
@test bij isa NamedTuple{(:a, :b, :c, :d)}
@test values(bij) == (bijector(KernelUniform()), bijector(KernelExponential()), bijector(KernelNormal()), bijector(KernelNormal()))

using Plots
plotly()
nt = rand(Random.default_rng(), d_mod)
histogram([rand(Random.default_rng(), d_mod).d for _ in 1:100])
histogram!([rand(Random.default_rng(), d).d for _ in 1:100])

@test logdensityof(d, nt) == logdensityof(d_mod, nt) - 1
# TODO not implemented
bij = bijector(d_mod)
@test bij isa NamedTuple{(:a, :b, :c, :d)}
@test values(bij) == (bijector(KernelUniform()), bijector(KernelExponential()), bijector(KernelNormal()), bijector(KernelNormal()))

######################## sequentialized graph ########################

sequentialize(node::AbstractNode) =
    traverse(node, (;)) do node, _
        node
    end

Base.rand(rng::AbstractRNG, graph::NamedTuple{<:Any,<:Tuple{Vararg{AbstractNode}}}, dims::Integer...) = rand_unroll(rng, values(graph), (;), dims...)
# unroll required for type stability
@unroll function rand_unroll(rng::AbstractRNG, graph, variables, dims::Integer...)
    @unroll for node in graph
        value = rand_barrier(node, variables, rng, dims...)
        variables = merge_value(variables, node, value)
    end
    variables
end

# still don't get why reduce(.+, map...) is type stable but mapreduce(.+,...) not
DensityInterface.logdensityof(graph::NamedTuple{names,<:Tuple{Vararg{AbstractNode}}}, nt::NamedTuple) where {names} =
    reduce(.+, map(values(graph), values(nt[names])) do node, value
        logdensityof(node(nt), value)
    end)


seq_graph = sequentialize(d)
rand(Random.default_rng(), seq_graph)
nt = @inferred rand(Random.default_rng(), seq_graph)
ℓ = @inferred logdensityof(seq_graph, nt)
@test ℓ == logdensityof(KernelUniform(), nt.a) + logdensityof(KernelExponential(), nt.b) + logdensityof(KernelNormal(nt.a, nt.b), nt.c) + logdensityof(KernelNormal(nt.c, nt.b), nt.d)


# WARN non-simplified implementations are not type stable
@benchmark rand(Random.default_rng(), d)
# 30x faster
@benchmark rand(Random.default_rng(), seq_graph)

@benchmark rand(Random.default_rng(), d, 100)
# 2x faster - gets less for larger dims
@benchmark rand(Random.default_rng(), seq_graph, 100)

# For now no automatic broadcasting of logdensityof
vars = rand(Random.default_rng(), seq_graph)
@benchmark logdensityof(d, vars)
# 30x faster
@benchmark logdensityof(seq_graph, vars)
