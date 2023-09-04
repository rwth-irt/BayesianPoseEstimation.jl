# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Plots
using Quaternions
using KernelDistributions
using MCMCDepth
using Rotations

# Works only with plotly, JS required for pdf export
plotlyjs()
diss_defaults()

# Euler non-uniform
eulers = [RotZYX((2π * rand(3))...) for _ in 1:500_000];
p_euler = sphere_density(eulers; n_θ=100, n_ϕ=50, xticks=[-1, 0, 1], yticks=[-1, 0, 1], zticks=[-1, 0, 1], colorbartitle=" probability density", title="Euler angles", legend=false)

# Quaternion uniform
quats = [QuatRotation(randn(QuaternionF32)) for _ in 1:500_000];
p_quat = sphere_density(quats; n_θ=100, n_ϕ=50, xticks=[-1, 0, 1], yticks=[-1, 0, 1], zticks=[-1, 0, 1], title="Quaternions", legend=false)

p = plot(p_euler, p_quat, size=MCMCDepth.correct_size((148.4789, 60)))
savefig(p, "plots/random_rotation.pdf")
