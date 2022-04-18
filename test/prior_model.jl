# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

# TODO autocomplete but no Revise?
# include("../src/MCMCDepth.jl")
# using .MCMCDepth
using MCMCDepth
using CUDA
using MeasureTheory

a_model = KernelProduct([KernelExponential(2.0f0), KernelExponential(1.0f0), KernelExponential(0.5f0)])
rand(a_model)
b_model = KernelExponential(2.0)
rand(b_model)
c_model = VectorizedMeasure(fill(KernelExponential(2.0), 3, 3))
rand(c_model, 2)

prior_model = IndependentPrior((; a=a_model, b=b_model, c=c_model))
s = rand(prior_model)
prior_model = IndependentPrior((; a=a_model, b=b_model, c=c_model))
s = rand(prior_model)
logdensity(prior_model, s)
# WARN granularity for CPU / GPU of different components? Use mutating version in performance critical code.
s_gpu = rand(CUDA.default_rng(), prior_model)
