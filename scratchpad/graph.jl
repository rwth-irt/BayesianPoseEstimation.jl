using Bijectors
using BenchmarkTools
using DensityInterface
using MCMCDepth
using Random
using Test
using Unrolled

abstract type AbstractNode{name,childnames} end

nodes(node::AbstractNode) = node.nodes

argvalues(::AbstractNode{<:Any,childnames}, nt) where {childnames} = values(nt[childnames])
varvalue(::AbstractNode{varname}, nt) where {varname} = nt[varname]

# realize the distribution
(node::AbstractNode)(nt::NamedTuple) = node.dist.(argvalues(node, nt)...)

# fn first to enable do syntax
function traverse(fn, node::AbstractNode{varname}, nt::NamedTuple{names}, args...) where {varname,names}
    # Termination: Value already available (conditioned on or calculate via another path)
    if varname in names
        return nt
    end
    # Conditional = values from other nodes required, compute depth first
    for n in nodes(node)
        nt = traverse(fn, n, nt, args...)
    end
    # Finally the internal dist can be realized and the value for this node can be merged
    retval = fn(node, nt, args...)
    merge(nt, NamedTuple{(varname,)}((retval,)))
end

simplify(node::AbstractNode) =
    traverse(node, (;)) do node, _
        node
    end

Base.rand(rng::AbstractRNG, node::AbstractNode{varname}, dims::Integer...) where {varname} =
    traverse(node, (;), rng, dims...) do node_, nt_, rng_, dims_...
        if isempty(nodes(node_))
            # Use dims only in leafs, otherwise they potentiate
            rand(rng_, node_(nt_), dims_...)
        else
            rand(rng_, node_(nt_))
        end
    end

DensityInterface.logdensityof(node::AbstractNode, nt::NamedTuple) =
    reduce(.+, traverse(node, (;)) do node_, _
        logdensityof(node_(nt), varvalue(node_, nt))
    end)

function Bijectors.bijector(node::AbstractNode)
    sample = rand(node)
    traverse(node, (;), sample) do node_, _...
        bijector(node_(sample))
    end
end

#################

# TODO Idea is that this node just wraps another one and thus shares the same varname and argnames
# TODO makes it harder to compile into static graph? - not really, simply merge from the right as (;varname=this_node)
struct ModifierNode{varname,argnames,M,N<:AbstractNode{varname,argnames}} <: AbstractNode{varname,argnames}
    model::M
    node::N
end

nodes(node::ModifierNode) = (node.node,)

##################

struct ConditionalNode{varname,names,D<:Function,N<:NamedTuple{names}} <: AbstractNode{varname,names}
    # Must be function to avoid UnionAll type instabilities
    dist::D
    nodes::N
end

ConditionalNode(varname::Symbol, dist::D, nodes::N) where {names,D<:Function,N<:NamedTuple{names}} = ConditionalNode{varname,names,D,N}(dist, nodes)

function ConditionalNode(varname::Symbol, ::Type{D}, nodes::N) where {names,D,N<:NamedTuple{names}}
    # Workaround so D is not UnionAll but interpreted as constructor
    wrapped = (x...) -> D(x...)
    ConditionalNode{varname,names,typeof(wrapped),N}(wrapped, nodes)
end

##################

a = ConditionalNode(:a, KernelUniform, (;))
b = ConditionalNode(:b, KernelExponential, (;))
c = ConditionalNode(:c, KernelNormal, (; a=a, b=b))
d = ConditionalNode(:d, KernelNormal, (; c=c, b=b))

nt = rand(Random.default_rng(), d)
ℓ = logdensityof(d, nt)
@test ℓ == logdensityof(KernelUniform(), nt.a) + logdensityof(KernelExponential(), nt.b) + logdensityof(KernelNormal(nt.a, nt.b), nt.c) + logdensityof(KernelNormal(nt.c, nt.b), nt.d)
bij = bijector(d)
@test bij isa NamedTuple{(:a, :b, :c, :d)}
@test values(bij) == (bijector(KernelUniform()), bijector(KernelExponential()), bijector(KernelNormal()), bijector(KernelNormal()))

########################

Base.rand(rng::AbstractRNG, graph::NamedTuple{<:Any,<:Tuple{Vararg{AbstractNode}}}, dims::Integer...) = rand_unroll(rng, values(graph), (;), dims...)
# unroll required for type stability
@unroll function rand_unroll(rng::AbstractRNG, graph, nt, dims::Integer...)
    @unroll for node in graph
        nt = rand_barrier(rng, node, nt, dims...)
    end
    nt
end
# Use dims only in leafs, otherwise they potentiate
rand_barrier(rng, node::AbstractNode{name,()}, nt, dims...) where {name} = (; nt..., name => rand(rng, node(nt), dims...))
rand_barrier(rng, node::AbstractNode{name}, nt, dims...) where {name} = (; nt..., name => rand(rng, node(nt)))

# still don't get why reduce(.+, map...) is type stable but mapreduce(.+,...) not
DensityInterface.logdensityof(graph::NamedTuple{names,<:Tuple{Vararg{AbstractNode}}}, nt::NamedTuple) where {names} =
    reduce(.+, map(values(graph), values(nt[names])) do node, value
        logdensityof(node(nt), value)
    end)

simplified = simplify(d)
rand(Random.default_rng(), simplified)
nt = @inferred rand(Random.default_rng(), simplified)
ℓ = @inferred logdensityof(simplified, nt)
@test ℓ == logdensityof(KernelUniform(), nt.a) + logdensityof(KernelExponential(), nt.b) + logdensityof(KernelNormal(nt.a, nt.b), nt.c) + logdensityof(KernelNormal(nt.c, nt.b), nt.d)


# WARN non-simplified implementations are not type stable
@benchmark rand(Random.default_rng(), d)
# 100x faster
@benchmark rand(Random.default_rng(), simplified)

@benchmark rand(Random.default_rng(), d, 10)
# 10x faster - gets negliable for larger dims
@benchmark rand(Random.default_rng(), simplified, 10)

# For now no automatic broadcasting of logdensityof
nt = rand(Random.default_rng(), simplified)
@benchmark logdensityof(d, nt)
# 100x faster
@benchmark logdensityof(simplified, nt)
