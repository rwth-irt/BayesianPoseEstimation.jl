# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 


"""
Incremental temperature schedules with the goal to sample from posterior for t → n (number of steps), thus select ϕ₀=0 → ϕₙ 1
"""

"""
    ConstantSchedule
Sample from the untempered posterior p(θ|z) ∝ p(z|θ)¹ p(θ)
"""
struct ConstantSchedule end

increment_temperature(schedule::ConstantSchedule, temperature) = 1

"""
    LinearSchedule(n_steps)
Linearly schedule the temperature ϕ from 0 to 1 over `n_steps`: p(z|θ)ᵠ p(θ)
"""
struct LinearSchedule
    n_steps::Float64
end

increment_temperature(schedule::LinearSchedule, temperature) = min(1, temperature + inv(schedule.n_steps))

"""
    LinearSchedule
Exponentially schedule the temperature ϕ from 0 to 1: p(z|θ)ᵠ p(θ)
"""
struct ExponentialSchedule
    λ::Float64
end

"""
    ExponentialSchedule(n_steps, goal_temp)
Create a schedule that will exponentially saturate to 1 and reach the `goal_temp` after `n_steps`.
"""
ExponentialSchedule(n_steps, goal_ϕ) = ExponentialSchedule(-log(1 - goal_ϕ) / n_steps)

increment_temperature(schedule::ExponentialSchedule, temperature) = temperature * exp(-schedule.λ) + 1 - exp(-schedule.λ)
