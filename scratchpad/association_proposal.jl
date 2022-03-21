# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using CUDA
using MCMCDepth
using MeasureTheory

#TODO strong typing
struct ImageModel
    gen_fn
    obs::AbstractMatrix
end

struct AssociationProposal
    o::AbstractArray{<:Real,3}
    μ::AbstractMatrix{<:Real}
end

pixel_factory(; μ, o, σ, λ, _...) = MCMCDepth.GpuBinaryMixture(MCMCDepth.GpuNormal(μ, σ), MCMCDepth.GpuExponential(λ), o, 1.0 - o)
gen_fn = pixel_factory | (; λ=1.5, σ=0.1) #|> Function
my_gen = gen_fn | (; μ=2.0) | :o
my_gen(0.1)

my_obs = rand(gen_fn(; o=0.5, μ=3.0), 3, 3)
my_μ = CUDA.ones(3, 3) * 3

img_model = ImageModel(gen_fn, my_obs)
# TODO raw state & transform_and_logjac
my_o = CUDA.rand(3, 3, 2)

ap = AssociationProposal(my_o, my_μ)

function MeasureTheory.logdensity(model::ImageModel, prop::AssociationProposal)
    L = Array{Function}(undef, size(model.obs))
    # TODO Broadcast on GPU?
    for i in eachindex(L)
        cond_gen = model.gen_fn | (; μ=prop.μ[i]) | :o
        obs_i = model.obs[i]
        L[i] = o -> logdensity(cond_gen(o), model.obs[i])
    end
    L
    # cuL = CuArray(L)
    # pixel_probs = map.(cuL, prop.o)
    # reduced_probs = reduce(+, pixel_probs; dims=(1, 2))
    # dropdims(reduced_probs; dims=(1, 2))
end

M = logdensity(img_model, ap)
D = map.(M, 0.5 * ones(3, 3, 2)) |> CuArray
logdensity.(D, my_obs)

M |> CuArray
map.(M, 0.5 * ones(3, 3, 2))

# WARN Make sure everything is strongly typed to the same type
bm_logdensity(μ::T, z::T, o::T) where {T<:Real} = logdensity(MCMCDepth.GpuBinaryMixture(MCMCDepth.GpuNormal(μ, 0.1), MCMCDepth.GpuExponential(1.0), o, 1.0f0 - o), z)
my_o = CUDA.rand(3, 3, 4)
bm_logdensity.(my_μ, my_obs, my_o)
