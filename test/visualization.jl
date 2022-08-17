# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using Accessors
using CUDA
using LinearAlgebra
using MCMCDepth
using Plots
using Random
using SciGL
using Test

pyplot()
params = MCMCDepth.Parameters()
render_context = RenderConte

# CvCamera like ROS looks down positive z
scene = Scene(params, render_context)
t = [0, 0, 1.5]
r = normalize!([1, 0, 0, 0])
p = to_pose(t, r, QuatRotation)
μ = render(render_context, scene, 1, p)

# Plot depth images and override some plot parameters
histogram(μ |> Array |> flatten)
plot_depth_img(μ)
plot_depth_img(μ; color_scheme=:cividis)
plot_depth_img(μ; reverse=false)

# Probability images
o = rand(KernelUniform(0.5f0, 1.0f0), 100, 100)
plot_prob_img(o)
plot_prob_img(o, clims=nothing)
