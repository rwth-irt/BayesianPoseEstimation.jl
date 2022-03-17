# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.
using CUDA
using GLAbstraction
using SciGL

"""
An image model is a Matrix of model generator functions with the size of a tile.
Thus, this image model must be mapped to each tile.
Assuming independence of the pixel, the log likelihood of each tile is the sum of each log likelihood.

Basically, these are tiled map-reductions.
"""
# TODO Threaded implementations?

"""
    reduce(op, A, tiles)
Applies the reduction operation to the tiles.
"""
function Base.reduce(op, A::AbstractMatrix, tiles::Tiles)
    res = Vector{Float64}(undef, length(tiles))
    for i = 1:length(tiles)
        tile_view = view_tile(A, tiles, i)
        res[i] = reduce(op, tile_view)
    end
    res
end

"""
    map(f, A, tiles)
Element wise mapping of the function matrix to each tile.
The function matrix must have the same size as one tile.
"""
function Base.map(f::AbstractMatrix{<:Function}, A::AbstractMatrix, tiles::Tiles)
    res = copy(A)
    for i = 1:length(tiles)
        tile_view = view_tile(res, tiles, i)
        tile_view .= map.(f, tile_view)
    end
    res
end

"""
    map!(f, A, tiles)
Element wise mapping of the function matrix to each tile.
The function matrix must have the same size as one tile.
"""
function Base.map!(f::AbstractMatrix{<:Function}, A::AbstractMatrix, tiles::Tiles)
    for i = 1:length(tiles)
        tile_view = view_tile(A, tiles, i)
        tile_view .= map.(f, tile_view)
    end
    A
end

"""
    mapreduce(f, op, A, tiles)
Element wise mapping of the function matrix to each tile.
The function matrix must have the same size as one tile.
The reduction operation reduces the Matrix to a vector of length(tiles).
"""
function Base.mapreduce(f::AbstractMatrix{<:Function}, op, A::AbstractMatrix, tiles::Tiles)
    mapped = map(f, A, tiles)
    reduce(op, mapped, tiles)
end

# Map reduction as tiled CUDA kernel

maybe_float32(::CuDeviceTexture, x) = Float32(x)
maybe_float32(::Any, x) = x
maybe_float32(::Val{true}, x) = Float32(x) # For testing

"""
    mapreduce(f, op, A, tiles, out)
CUDA kernel for element wise mapping of the function matrix to each tile in A.
The function matrix must have the same size as one tile.
The reduction operation reduces the Matrix to a vector of length(n_blocks).

Choose: length(out) = block_size = N, shmem = n_threads * sizeof(T)
"""
function Base.mapreduce(f::CuDeviceMatrix{<:Function}, op, A::Union{CuDeviceMatrix,CuDeviceTexture}, tiles::Tiles, out::CuDeviceVector{T}) where {T}
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

"""
    mapreduce(f, op, A, tiles, n_threads)
CUDA kernel for element wise mapping of the function matrix to each tile.
The function matrix must have the same size as one tile.
The reduction operation reduces the Matrix to a vector of length(tiles).
"""
function Base.mapreduce(f::CuMatrix{<:Function}, op, A::Union{CuMatrix,CuTexture}, tiles::Tiles, n_threads = 256)
    n_blocks = length(tiles)
    out = CuVector{Float32}(undef, n_blocks)
    shmem_size = n_threads * sizeof(eltype(out))
    CUDA.@cuda blocks = n_blocks threads = n_threads shmem = shmem_size mapreduce(f, op, A, tiles, out)
    out
end

"""
    mapreduce(f, op, A, tiles, n_threads)
CUDA kernel for element wise mapping of the function matrix to each tile.
The function matrix must have the same size as one tile.
The reduction operation reduces the Matrix to a vector of length(tiles).
"""
Base.mapreduce(f::AbstractMatrix{<:Function}, op, A::Union{CuMatrix,CuTexture}, tiles::Tiles, n_threads = 256) = mapreduce(CuArray(f), op, A, tiles, n_threads)

"""
    mapreduce(f, op, A, tiles, n_threads)
CUDA kernel for element wise mapping of the function matrix to each tile.
The function matrix must have the same size as one tile.
The reduction operation reduces the Matrix to a vector of length(tiles).
"""
Base.mapreduce(f::AbstractMatrix{<:Function}, op, A::GLAbstraction.Texture, tiles::Tiles, n_threads = 256) = mapreduce(f, op, A, CuTexture(tiles), n_threads)

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
