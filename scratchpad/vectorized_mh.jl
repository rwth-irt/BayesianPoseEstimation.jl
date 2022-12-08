include("../src/MCMCDepth.jl")
using .MCMCDepth

using CUDA
using EllipsisNotation
using MCMCDepth
using Random

rng = CUDA.default_rng()

D = 2
N = 3

logp = rand(rng, N) .|> log
prev_logp = rand(rng, N) .|> log
forward = rand(rng, N) .|> log
backward = rand(rng, N) .|> log

# NOTE Using broadcast instead allows this operation to be compiled into a single kernel
α = logp .- prev_logp .+ backward .- forward
reject = log.(rand(rng, N)) .> α

prev_sample = rand(rng, D, N)
new_sample = rand(rng, D, N)
merged_sample = similar(prev_sample)
CUDA.@time @views merged_sample[.., reject] = prev_sample[.., reject]
CUDA.@time merged_sample[.., map(!, reject)] = new_sample[.., map(!, reject)]

# NOTE assumes that only last dim is vectorized
function vectorized_mh(rng::AbstractRNG, proposal::Proposal, proposed::Sample, previous::Sample)
    # Using broadcast instead allows this operation to be compiled into a single kernel
    α = (logprob(proposed) .-
         logprob(previous) .+
         transition_probability(proposal, previous, proposed) .-
         transition_probability(proposal, proposed, previous)
    )
    reject = log.(rand(rng, N)) .> α
end

accept_reject_vectorized!(reject::AbstractVector{Bool}, proposed::AbstractArray{T,N}, previous::AbstractArray{T,N}) where {T,N} = @views proposed[.., reject] = previous[.., reject]

# TODO test if @views works for inline function

reject
merged_sample
prev_sample
cpy = copy(new_sample)
accept_reject_vectorized!(reject, cpy, prev_sample)
cpy == merged_sample
