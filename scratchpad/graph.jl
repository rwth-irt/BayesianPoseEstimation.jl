using Bijectors
using BenchmarkTools
using DensityInterface
using MCMCDepth
using Random
using Test

struct ConditionalNode{varname,names,D<:Function,N<:NamedTuple{names}}
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

argvalues(::ConditionalNode{<:Any,names}, nt) where {names} = values(nt[names])
varvalue(::ConditionalNode{varname}, nt) where {varname} = nt[varname]

# realize the distributioncnames,
(node::ConditionalNode)(x...) = node.dist.(x...)
# (node::ConditionalNode)(; y...) = node.dist(argvalues(node, y)...)
(node::ConditionalNode)(nt::NamedTuple) = node.dist.(argvalues(node, nt)...)

# fn first to enable do syntax
function traverse(fn, node::ConditionalNode{varname}, nt::NamedTuple{names}, args...) where {varname,names}
    # Termination: Value already available (conditioned on or calculate via another path)
    if varname in names
        return nt
    end
    # Conditional = values from other nodes required, compute depth first
    for n in node.nodes
        # TODO modification of nt seemingly make this instable, but keeping it constant does not change runtime
        # WARN wrong implementation traverse(fn, n, nt, args...)
        nt = traverse(fn, n, nt, args...)
    end
    # Finally the internal dist can be realized and the value for this node can be merged
    retval = fn(node, nt, args...)
    merge(nt, NamedTuple{(varname,)}((retval,)))
end

Base.rand(rng::AbstractRNG, node::ConditionalNode{varname}, dims::Integer...) where {varname,names} =
    traverse(node, (;), rng, dims...) do node_, nt_, rng_, dims_...
        if isempty(node_.nodes)
            rand(rng_, node_(nt_), dims_...)
        else
            rand(rng_, node_(nt_))
        end
    end

DensityInterface.logdensityof(node::ConditionalNode, nt::NamedTuple) =
    reduce(.+, traverse(node, (;)) do node, _
        logdensityof(node(nt), varvalue(node, nt))
    end)

function Bijectors.bijector(node::ConditionalNode)
    sample = rand(node)
    traverse(node, (;), sample) do node_, _...
        bijector(node_(sample))
    end
end

a = ConditionalNode(:a, KernelUniform, (;))
b = ConditionalNode(:b, KernelExponential, (;))
c = ConditionalNode(:c, KernelNormal, (; a=a, b=b))
d = ConditionalNode(:d, KernelNormal, (; c=c, b=b))

nt = rand(Random.default_rng(), d)
ℓ = logdensityof(d, nt)
@test ℓ == logdensityof(KernelUniform(), nt.a) + logdensityof(KernelExponential(), nt.b) + logdensityof(KernelNormal(nt.a, nt.b), nt.c) + logdensityof(KernelNormal(nt.c, nt.b), nt.d)

# WARN non-generic implementations are faster

function rand_specialized(rng::AbstractRNG, node::ConditionalNode{varname}, nt::NamedTuple{names}, dims::Integer...) where {varname,names}
    if varname in names
        return nt
    end
    for n in node.nodes
        nt = rand(rng, n, nt, dims...)
    end
    # Only apply dims for leafs
    if isempty(node.nodes)
        retval = rand(rng, node(nt), dims...)
    else
        retval = rand(rng, node(nt))

    end
    merge(nt, NamedTuple{(varname,)}((retval,)))
end

function logdensityof_specialized(node::ConditionalNode{varname}, nt::NamedTuple{names}, obs) where {varname,names}
    if varname in names
        return nt
    end
    for n in node.nodes
        nt = logdensityof(n, nt, obs)
    end
    retval = logdensityof(node(obs), varvalue(node, obs))
    merge(nt, NamedTuple{(varname,)}((retval,)))
end

nt = @btime rand(Random.default_rng(), c)
nt = @btime rand_specialized(Random.default_rng(), c, (;))

ℓ = @btime logdensityof(c, nt)
ℓ = @btime logdensityof_specialized(c, (;), nt)
