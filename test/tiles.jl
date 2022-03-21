# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.
using BenchmarkTools
using MCMCDepth
using SciGL

# Test the GPU implementation against the CPU implementation
tiles = Tiles(2, 2, 6, 2, 3)
M = rand(size(tiles)...)
using Distributions
dist_nt = (; fn=(x) -> logpdf(Normal(0.0, 1.0), x), dist=Normal(0.0, 1.0))
fn = dist_nt.fn
models = fill(nt.fn, tile_size(tiles))
res_cpu = mapreduce(models, +, M, tiles)
# get_property and getindex for nt fails
function gpureduce(nt, data)
    cuM = CuArray(data)
    cumodels = CUDA.fill(x -> logpdf(nt.dist, x), tile_size(tiles))
    mapreduce(cumodels, +, cuM, tiles)
end
gpureduce(dist_nt, M)

res_cpu ≈ Array(res_gpu)

bench_tile_1 = Tiles(100, 100, 1, 1, 1)
bench_M_1 = rand(size(bench_tile_1)...)
bench_cuM_1 = CuArray(bench_M_1)
bench_f_1 = fill(exp, tile_size(bench_tile_1))
@benchmark mapreduce(bench_f_1, +, Array(bench_cuM_1), bench_tile_1)
@benchmark CUDA.@sync mapreduce(bench_f_1, +, bench_cuM_1, bench_tile_1)
# Results similar, CUDA a bit fast. Guess it does not make a difference in which direction we copy

bench_tile_10000 = Tiles(100, 100, 10000, 100, 100)
bench_M_10000 = rand(size(bench_tile_10000)...)
bench_cuM_10000 = CuArray(bench_M_10000)
bench_f_10000 = fill(exp, tile_size(bench_tile_10000))
@benchmark mapreduce(bench_f_10000, +, Array(bench_cuM_10000), bench_tile_10000)
@benchmark CUDA.@sync mapreduce(bench_f_10000, +, bench_cuM_10000, bench_tile_10000)
# Way faster on GPU by avoiding copying the large array. Mapping the Texture to CUDA might offer even more benefits.
