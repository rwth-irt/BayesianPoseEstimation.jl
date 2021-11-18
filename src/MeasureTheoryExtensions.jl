# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using MeasureTheory
using TransformVariables

"""
    UniformInterval(a, b)
The *continuous uniform distribution* over an interval ``[a, b]`` has probability density function
```math
f(x; a, b) = \\frac{1}{b - a}, \\quad a \\le x \\le b
```
"""

@parameterized UniformInterval(a, b)

function MeasureTheory.basemeasure(d::UniformInterval{(:a, :b)})
    inbounds(x) = d.a <= x <= d.b
    constℓ = 0.0
    varℓ() = -log(d.b - d.a)
    base = Lebesgue(ℝ)
    FactoredBase(inbounds, constℓ, varℓ, base)
end

MeasureTheory.distproxy(d::UniformInterval{(:a, :b)}) = Dists.Uniform(d.a, d.b)

MeasureTheory.logdensity(d::UniformInterval{(:a, :b)}, x) = -log(d.b - d.a)

TransformVariables.as(d::UniformInterval{(:a, :b)}) = as(Real, d.a, d.b)

"""
    CircularUniform
Similar to the UniformInterval distribution but fixed to the interval [0,2π].
Uses the CircularTransform for continuity.
```
"""

@parameterized CircularUniform()

function MeasureTheory.basemeasure(::CircularUniform{()})
    inbounds(x) = 0 <= x <= 2π
    constℓ = -log(2π)
    varℓ() = 0.0
    base = Lebesgue(ℝ)
    FactoredBase(inbounds, constℓ, varℓ, base)
end

MeasureTheory.distproxy(::CircularUniform{()}) = Dists.Uniform(0, 2π)

MeasureTheory.logdensity(::CircularUniform{()}, x) = -log(2π)

TransformVariables.as(::CircularUniform{()}) = as○
