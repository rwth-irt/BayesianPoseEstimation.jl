# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using MCMCDepth
using MeasureTheory

a_model = GpuProductMeasure(GpuExponential(2), 3, 3)
a_var = SampleVariable(Float16, a_model)
SampleVariable(a_var)
MCMCDepth.logdensity_var(a_model, a_var)
b_model = For(2) do _
    Exponential(2.0)
end
tr = as(b_model)
b_var = SampleVariable(b_model)
MCMCDepth.logdensity_var(b_model, b_var)
ModelVariable(b_var)
c_model = Exponential(2.0)
c_var = SampleVariable(c_model)
MCMCDepth.logdensity_var(c_model, c_var)
d_model = VectorizedMeasure(Exponential(2.0), 3, 5)
d_var = SampleVariable(d_model)
MCMCDepth.logdensity_var(d_model, d_var)
e_model = GpuVectorizedMeasure(GpuExponential(2.0), 3, 5)
e_var = SampleVariable(e_model)
MCMCDepth.logdensity_var(e_model, e_var)
