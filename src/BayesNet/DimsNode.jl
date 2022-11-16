# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

# TODO rename dims to sizes?
# TODO Integrate into BroadcastedNode
struct DimsNode{name,child_names,C<:NamedTuple{child_names},R<:AbstractRNG,M<:Union{Distribution,Function},N,D<:Tuple{Vararg{Dims}}} <: AbstractNode{name,child_names}
    children::C
    rng::R
    # Must be function to avoid UnionAll type instabilities
    model::M
    model_dims::Dims{N}
    child_dims::D
end

# Convenience constructor for moving name to the parametric type
DimsNode_(name::Symbol, children::C, rng::R, model::M, dims::Dims{N}, child_dims::D) where {child_names,C<:NamedTuple{child_names},R<:AbstractRNG,M<:Union{Distribution,Function},N,D<:Tuple{Vararg{Dims}}} = DimsNode{name,child_names,C,R,M,N,D}(children, rng, model, dims, child_dims)


# WARN Manipulated function not type stable for Type as arg
# TODO This hides a lot of the logic
# broadcast_model(::Type{T}, dist_dims::Dims, child_dims) where {T} = (x...) -> BroadcastedDistribution.((T,), (dist_dims,), slicedims.(x, child_dims)...)
# broadcast_model(fn::Function, dist_dims::Dims, child_dims) = (x...) -> BroadcastedDistribution.((fn,), (dist_dims,), slicedims.(x, child_dims)...)

# TODO need to override logdensityof_barrier because logdensityof needs to be broadcasted

# slicedims(x::Real) = x
# # TODO this only works for single dims vectorization
# # TODO wait for fix of eachslice https://github.com/JuliaLang/julia/issues/39639
# slicedims(x::AbstractArray{<:Any,N}) where {N} = [selectdim(x, N, i) for i in 1:last(size(x))]
# # TODO Need to store the original dims of each parameter when using this
# # Real values and initialization with unknown dims
# slicedims(x, ::Val{0}) where {N} = x
# # Case of single sample
# # TODO still need to wrap this in another array when broadcasting logdensityof
# # slicedims(x::AbstractArray{<:Any,N}, ::Val{N}) where {N} = x
# function slicedims(x::AbstractArray{<:Any,N}, ::Val{D}) where {N,D}
#     res_size = size(x)[D+1:N]
#     res_type = view(x, ntuple(_ -> :, D)..., ntuple(_ -> 1, N - D)...) |> typeof
#     res = Array{res_type}(undef, res_size)
#     for i in CartesianIndices(res)
#         res[i] = view(x, ntuple(_ -> :, D)..., Tuple(i)...)
#     end
#     res
# end

# TEST e.g.: inflatedims(rand(2,3), (2,), (1,1,2)) orig. scalar: inflatedims(rand(2,3),(),(1,2))
"""
    insertdims(A, child_dims, dist_dims)
Reshape `A` by inserting dims of length one to make it compatible for broadcasting multiple random samples of differently shaped children.
`child_dims` are the dims of a single sample from the child node and `dist_dims` the dims of the broadcasted distribution which results from a single sample of all child nodes.

# Rationale
When proposing multiple samples, originally matching dims of the BroadcastedDistribution do not work anymore.
E.g. if one child has (3,) and the other (3,2) sized samples, proposing multiple samples result in incompatible dimensions (3,5) and (3,2,5).
Julia expands dimensions of length one when broadcasting, so reshaping the array with dimensions of length one enables proposing multiple samples, for the above: (3,5) insertdims → (3,1,5) broadcast → (3,2,5)
https://freecontent.manning.com/vectorizing-your-code-using-broadcasting/
https://discourse.julialang.org/t/designating-the-axes-for-broadcasting/29203/2
"""
insertdims(A::Real, ::Dims, ::Dims) = A
# TODO for initialization, does it cause unexpected behavior if the BroadcastedDistribution is actually scalar? Probably not since A must be a real in this case, which calls the above.
insertdims(A, ::Dims, ::Dims{0}) = A
function insertdims(A, original::Dims{O}, ::Dims{B}) where {O,B}
    # WARN fill array is not type stable
    fill_ones = ntuple(_ -> 1, B - O)
    # Dimension of multiple samples
    remaining = size(A)[O+1:end]
    reshape(A, original..., fill_ones..., remaining...)
end

function childvalues(node::DimsNode{<:Any,childnames}, nt::NamedTuple) where {childnames}
    child_values = values(nt[childnames])
    insertdims.(child_values, node.child_dims, (node.model_dims,))
end

# Construct as parent
"""
    DimsNode(name, children, rng, distribution)
Construct a node which automatically broadcasts the `distribution` over the parameters given by the `children`.
The resulting `BroadcastedDistribution` acts like a product distribution, reducing the ndims for the minimal realization of the distribution given the `children`.
"""
function DimsNode(name::Symbol, children::NamedTuple, rng::AbstractRNG, distribution::Callable)
    # Workaround so D is not UnionAll but interpreted as constructor
    # no reduction or slicing for initialization with single random value

    sacrifice_model = broadcast_model(distribution, ())
    sacrifice_child_dims = ntuple(_ -> (), length(children))
    sacrifice_node = DimsNode_(name, children, rng, sacrifice_model, (), sacrifice_child_dims)
    sacrifice_nt = rand(sacrifice_node)

    model_dims = param_dims(varvalue(sacrifice_node, sacrifice_nt))
    child_dims = size.(childvalues(sacrifice_node, sacrifice_nt))
    node_model = broadcast_model(distribution, model_dims)
    DimsNode_(name, children, rng, node_model, model_dims, child_dims)
end

# Construct as leaf
# TODO Do I like the parameter order, i.e. rng between params and distribution? Should model / distribution be moved to the end?
"""
    BroadcastedNode(name, rng, distribution, params...)
Construct the node as leaf (no children) by broadcasting the `distribution` over the `params`.
The resulting `BroadcastedDistribution` acts like a product distribution, reducing the ndims of the `params`.
"""
DimsNode(name::Symbol, rng::AbstractRNG, distribution, params...) = DimsNode_(name, (;), rng, ProductBroadcastedDistribution(distribution, params...), param_dims(params...), ())

# TODO remove?
# Leaf
# rand_barrier(node::DimsNode{<:Any,()}, variables::NamedTuple, dims...) = rand(rng(node), node(variables))
# Parent
# rand_barrier(node::DimsNode, variables::NamedTuple, dims...) = node.fn(childvalues(node, variables)...)
# logdensityof_barrier(node::DimsNode, variables::NamedTuple) = logdensityof.(node(variables), varvalue(node, variables))