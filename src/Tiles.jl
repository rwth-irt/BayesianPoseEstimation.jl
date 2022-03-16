# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

using CUDA
using SciGL

"""
An image model is a Matrix of model generator functions with the size of a tile.
Thus, this image model must be mapped to each tile.
Assuming independence of the pixel, the log likelihood of each tile is the sum of each log likelihood.

Basically, these are tiled map-reductions.
"""
# TODO Threaded implementations?

"""
    reduce(op, A, tiles, n)
Applies the reduction operation to `n` tiles.
Returns a vector of size `n`.
"""
function Base.reduce(op, A::AbstractMatrix, tiles::Tiles, n::Integer = length(tiles))
    res = Vector{Float64}(undef, n)
    for i = 1:n
        tile_view = view_tile(A, tiles, i)
        res[i] = reduce(op, tile_view)
    end
    res
end

"""
    map(f, A, tiles, n)
Element wise mapping of the function matrix to each tile.
The function matrix must have the same size as one tile.
"""
function Base.map(f::AbstractMatrix{<:Function}, A::AbstractMatrix, tiles::Tiles, n::Integer = length(tiles))
    res = copy(A)
    for i = 1:n
        tile_view = view_tile(res, tiles, i)
        tile_view .= map.(f, tile_view)
    end
    res
end

"""
    map!(f, A, tiles, n)
Element wise mapping of the function matrix to each tile.
The function matrix must have the same size as one tile.
"""
function Base.map!(f::AbstractMatrix{<:Function}, A::AbstractMatrix, tiles::Tiles, n::Integer = length(tiles))
    for i = 1:n
        tile_view = view_tile(A, tiles, i)
        tile_view .= map.(f, tile_view)
    end
    A
end

"""
    mapreduce(f, op, A, tiles, n)
Element wise mapping of the function matrix to each tile.
The function matrix must have the same size as one tile.
The reduction operation reduces the Matrix to a vector of length n.
"""
function Base.mapreduce(f::AbstractMatrix{<:Function}, op, A::AbstractMatrix, tiles::Tiles, n::Integer = length(tiles))
    mapped = map(f, A, tiles, n)
    reduce(op, mapped, tiles, n)
end

# Map reduction as tiled CUDA kernel

maybe_float32(::CuDeviceTexture, x) = Float32(x)
maybe_float32(::Any, x) = x
maybe_float32(::Val{true}, x) = Float32(x) # For testing

"""
    mapreduce(f, op, A, tiles, out)
Element wise mapping of the function matrix to each tile.
The function matrix must have the same size as one tile.
The reduction operation reduces the Matrix to the vector out.

Choose: length(out) = block_size = N, shmem = n_threads * sizeof(T
"""
function Base.mapreduce(f::AbstractArray{<:Function}, op, A::AbstractMatrix, tiles::Tiles, out::CuDeviceVector{T}) where {T}
    # aliases for readability
    thread_id = threadIdx().x
    block_id = blockIdx().x
    n_threads = blockDim().x
    # Thread local accumulation
    thread_acc = zero(T)
    # thread strided loop
    for i in thread_id:n_threads:tile_length(tiles)
        # Texture indices: N-Dims Float32
        x, y = maybe_float32.((A,), coordinates(tiles, block_id, i))
        thread_acc = op(thread_acc, f[i](A[x, y]))
    end
    # Synchronized accumulation for block
    block_acc = CuDynamicSharedArray(Float32, n_threads)
    @inbounds block_acc[thread_id] = thread_acc
    sync_threads()
    if thread_id == 1
        @inbounds out[block_id] = reduce(op, block_acc)
    end
    return nothing
end

# Test the GPU implementation against the CPU implementation
M = rand(4, 6)
models = fill(x -> 2 * x, 2, 2)
tiles = Tiles(2, 2, 6, 2, 3)
res_cpu = mapreduce(models, +, M, tiles)

cum = CuArray(M)
cumodels = CuArray(models)
res_gpu = CuVector{Float32}(undef, length(tiles))
n_blocks = length(res_gpu)
n_threads = 256
shmem_size = n_threads * sizeof(eltype(res_gpu))
CUDA.@cuda blocks = n_blocks threads = n_threads shmem = shmem_size mapreduce(cumodels, +, cum, tiles, res_gpu)

res_cpu ≈ Array(res_gpu)
