# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

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
gl_context = depth_offscreen_context(params.width, params.height, params.depth, Array)

@reset params.mesh_file = "meshes/BM067R.obj"
scene = Scene(gl_context, params)
# CvCamera like ROS looks down positive z
t = [-0.05, 0.05, 0.25]
r = [1, 1, 0]
p = to_pose(t, r)
μ = render(gl_context, scene, p)

# Plot depth images and override some plot parameters
plot_depth_img(μ)
plot_depth_img(μ; color_scheme=:cividis)
plot_depth_img(μ; reverse=false)
histogram(μ |> Iterators.flatten |> collect)

# Probability images
o = rand(KernelUniform(0.5f0, 1.0f0), 100, 100)
plot_prob_img(o, value_to_typemax=maximum(o))
plot_prob_img(o, clims=nothing)
