# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using MCMCDepth
using MeasureTheory

a_model = GpuProductMeasure(GpuExponential(2), 3, 3)
b_model = For(2) do _
    Exponential(2.0)
end
c_model = Exponential(2.0)
d_model = VectorizedMeasure(Exponential(2.0), 3, 5)
e_model = GpuVectorizedMeasure(GpuExponential(2.0), 3, 5)

prior_model = IndependentPrior((;a = a_model, b = b_model, c = c_model, d = d_model, e = e_model))

s = rand(prior_model)
