# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# TODO always sample in constrained state? guess independent sampling should be in constrained space by default?
# TODO is it possible to infer it from the function or should I actually make a new proposal type
# function f_sample_proposal(s::AbstractSample, f::Function)
#     θ = f(state(s))
#     s_new = copy(s)
#     state!(s_new, θ)
# end

# TODO alternatively: Dispatch depends on type 
# function f_sample_proposal(s::T, f::Function)
#     θ = f(unconstrained_state(s))
#     s_new = copy(s)
#     unconstrained_state!(s_new, θ)
# end


"""
    mh_step(s, f, q, ℓ)
Metropolis-Hastings step which returns the next sample of the chain based on the previous sample `s`.
The function `f(s)` is used to propose a new sample with the probability `q(s,s_cond)=q(s|s_cond)`.
Both samples are evaluated using the likelihood function `ℓ(s)`
"""
function mh_step(s::T, f::Function, q::Function, ℓ::Function; rng::AbstractRNG = Random.GLOBAL_RNG) where {T}
    # propose new sample
    s_new = f(s)
    # acceptance ratio
    # TODO use likelihood*prior or joint (=l*p)
    α = ℓ(s_new) - ℓ(s) + q(s, s_new) - q(s_new, s)
    if log(rand(rng)) > α
        # reject
        return s
    else
        # accept (always the case if difference positive since log([0,1])->[-inf,0])
        return s_new
    end
end
