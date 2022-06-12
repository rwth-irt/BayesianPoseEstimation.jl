# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Bijectors
using MCMCDepth
using Test

# Create and convert ModelVariables
amv = ModelVariable([0.5, 0.9], bijector(Uniform()))
@test model_value(amv)[1] == 0.5
# logit(0.5) = 0.0
@test raw_value(amv)[1] == 0.0
asv = convert(SampleVariable, amv)
@test model_value(asv)[1] == 0.5
@test raw_value(asv)[1] == 0.0
bsv = SampleVariable(1, bijector(Exponential()))
bmv = ModelVariable(bsv)

# Model Variables add in model domain
@test model_value(amv + amv) == [1.0, 1.8]
@test model_value(amv - amv) == [0, 0]
# Sample Variables add in unconstrained domain
@test raw_value(bsv + bsv) == 2
@test raw_value(bsv - bsv) == 0
# Mixed variables in unconstrained by default
@test raw_value(bmv + bsv) == 2
@test raw_value(bmv + bsv) == 2
# minus mixed
@test raw_value(bsv - bmv) == 0
@test raw_value(bmv - bsv) == 0

# TODO test
a = @inferred asv + bsv
b = @inferred bsv + asv
c = @inferred amv + bmv
d = @inferred amv + bsv
@test raw_value(a) == raw_value(b)

# Sample
nta = (; zip((:a, :b), fill(amv, 2))...)
ntb = (; zip((:b, :c), fill(bsv, 2))...)
sa = Sample(nta, 0.0)
sb = Sample(ntb, 0.0)
@test model_value(variables(@inferred sa + sb)[1]) == [0.5, 0.9]
@test raw_value(variables(@inferred sa + sb)[2]) == [1.0, 3.1972245773362196]
@test model_value(variables(@inferred sa - sb)[1]) == [0.5, 0.9]
@test raw_value(variables(@inferred sa - sb)[2]) == [-1.0, 1.1972245773362196]
