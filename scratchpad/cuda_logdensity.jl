# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 
using BenchmarkTools
using CUDA
using Soss, MeasureTheory

# Observations: All of the implementations perform equally well during the computation.
# Most of the time is spent on copying the data to the GPU.
# More complex Array types take longer to transfer
# Passing conditioned functions as Matrix which only require μ (mean from rendered image) is flexible and quite fast.
# Since length(μ) ≫ length(z) = length(fns) the overhead should be negligible.

N = 100000
n_threads = 256
n_blocks = div(N, n_threads, RoundUp)
z = rand(Float32, N)
cu_z = CuArray(z)
μ = CUDA.rand(N)
out = CUDA.zeros(N)

function hardcoded(μ, z, out)
  index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  stride = gridDim().x * blockDim().x
  for i = index:stride:length(z)
    @inbounds out[i] = logdensity(Normal(μ[i], 1.0), z[i])
  end
  nothing
end

# Best speed is just passing the matrices and the function definition
function mat_fn(fn, μ, z, out)
  index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  stride = gridDim().x * blockDim().x
  for i = index:stride:length(z)
    @inbounds out[i] = fn(μ[i], z[i])
  end
  nothing
end

# TODO best tradeoff between speed and flexibility
function mat_fns(fns, μ, out)
  index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  stride = gridDim().x * blockDim().x
  for i = index:stride:length(μ)
    @inbounds out[i] = fns[i](μ[i])
  end
  nothing
end
out2 = CUDA.zeros(N)


# WARN needs to be strongly typed for CUDA kernel
struct FnParam
  z::Float32
  σ::Float32
end

function mat_pfn(pfn, μ, p, out)
  index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  stride = gridDim().x * blockDim().x
  for i = index:stride:length(p)
    @inbounds out[i] = pfn(μ[i], p[i])
  end
  nothing
end

@benchmark begin
  # Hard coded is not faster than passing the function
  # However, copying the data is what takes most of the time
  copyto!(cu_z, z)
  CUDA.@sync @cuda threads = n_threads blocks = n_blocks hardcoded(μ, cu_z, out)
end

fn(μ, z) = logdensity(Normal(μ, 1.0), z)
@benchmark begin
  # WARN Avoid out of GPU memory by pre-allocating
  copyto!(cu_z, z)
  CUDA.@sync @cuda threads = n_threads blocks = n_blocks mat_fn(fn, μ, cu_z, out)
end

@benchmark begin
  # WARN Can be horribly slow if we condition on CuArrays (lots of copies)
  fns = [μ -> logdensity(Normal(μ, 1.0,), z_i) for z_i in z]
  cu_fns = CuArray(fns)
  CUDA.@sync @cuda threads = n_threads blocks = n_blocks mat_fns(cu_fns, μ, out)
end

pfn(μ, p::FnParam) = logdensity(Normal(μ, 1.0), p.z)
@benchmark begin
  params = FnParam.(z, (1.0))
  cu_params = CuArray(params)
  CUDA.@sync @cuda threads = n_threads blocks = n_blocks mat_pfn(pfn, μ, cu_params, out)
end

