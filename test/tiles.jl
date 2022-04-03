# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.
using BenchmarkTools
using CUDA
using MCMCDepth
using SciGL

"""
    mapreduce12(f, op, A, idx)
Map reduce by first rearranging A to a 3D array via idx.
"""
function mapreduce12(f, op, A, idx)
    # About 10ms faster than without lazy, avoids a whole Array allocation, since CUDA.jl has special implementations for Broadcasted, which avoid additional allocations
    # TODO Other allocation happens at A[idx]
    lazy_map = Broadcast.broadcasted(map, f, A[idx])
    # CUDA takes care of materializing the lazy broadcasted
    dropdims(reduce(op, lazy_map, dims=(1, 2)), dims=(1, 2))
end

# Test if CPU implementation equals GPU implementation
tiles = Tiles(2, 2, 6, 2, 3)
M = rand(size(tiles)...)
fn(x) = exp(x) + x
models = fill(dist_nt.fn, tile_size(tiles))
res_cpu = mapreduce(models, +, M, tiles)
res_gpu = mapreduce(CuArray(models), +, CuArray(M), tiles)
@assert res_cpu ≈ Array(res_gpu)
# WARN this must be pre-calculated and stored on the GPU, otherwise it is really slow
idx = LinearIndices(tiles) |> CuArray
res_gpu12 = mapreduce12(CuArray(models), +, CuArray(M), idx)
@assert res_cpu ≈ Array(res_gpu12)

# Benchmarks
bench_tile_1 = Tiles(100, 100, 1, 1, 1)
bench_idx_1 = LinearIndices(bench_tile_1)
bench_cuidx_1 = CuArray(bench_idx_1)
bench_M_1 = rand(size(bench_tile_1)...)
bench_cuM_1 = CuArray(bench_M_1)
bench_f_1 = fill(exp, tile_size(bench_tile_1))
# 141.135 μs ± 92.717 μs 
@benchmark mapreduce(bench_f_1, +, Array(bench_cuM_1), bench_tile_1)
# 137.163 μs ±  11.112 μs  
@benchmark CUDA.@sync mapreduce(bench_f_1, +, bench_cuM_1, bench_tile_1)
# 79.795 μs ±  32.046 μs
@benchmark CUDA.@sync mapreduce12(CuArray(bench_f_1), +, bench_cuM_1, bench_cuidx_1)
# Results similar, CUDA a bit fast. Guess it does not make a difference in which direction we copy

bench_tile_5000 = Tiles(100, 100, 5000, 100, 50)
bench_idx_5000 = LinearIndices(bench_tile_5000)
bench_cuidx_5000 = CuArray(bench_idx_5000)
bench_M_5000 = rand(size(bench_tile_5000)...)
bench_cuM_5000 = CuArray(bench_M_5000)
bench_f_5000 = fill(exp, tile_size(bench_tile_5000))

# 638.457 ms ±  20.736 ms 
@benchmark mapreduce(bench_f_5000, +, bench_M_5000, bench_tile_5000)
# 78.235 ms ± 135.626 μs, 1 GPU allocation: 19.531 KiB,
@benchmark CUDA.@sync mapreduce(bench_f_5000, +, bench_cuM_5000, bench_tile_5000)
# 135.122 ms ±  40.899 ms, 2 GPU allocations: 381.508 MiB (3 GPU ≈ 760MB for non lazy version)
@benchmark CUDA.@sync mapreduce12(CuArray(bench_f_5000), +, bench_cuM_5000, bench_cuidx_5000)

bench_tile_10000 = Tiles(100, 100, 10000, 100, 100)
bench_idx_10000 = LinearIndices(bench_tile_10000)
bench_cuidx_10000 = CuArray(bench_idx_10000)
bench_M_10000 = rand(size(bench_tile_10000)...)
bench_cuM_10000 = CuArray(bench_M_10000)
bench_f_10000 = fill(exp, tile_size(bench_tile_10000))
@benchmark mapreduce(bench_f_10000, +, Array(bench_cuM_10000), bench_tile_10000)
@benchmark CUDA.@sync mapreduce(bench_f_10000, +, bench_cuM_10000, bench_tile_10000)
# Out of GPU Memory, while the custom kernel still works, see allocation notes in mapreduce12
@benchmark CUDA.@sync mapreduce12(CuArray(bench_f_10000), +, bench_cuM_10000, bench_cuidx_10000)

# Way faster on GPU by avoiding copying the large array. Mapping the Texture to CUDA might offer even more benefits.

# TODO Threaded implementation by default?
# TODO benchmark reindexing with the linear indices, implement reindex?
# TODO Expect 0.002 sec for copying from OpenGL to CUDA, if CuArray is pre-allocated
