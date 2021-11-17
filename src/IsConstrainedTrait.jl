# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Soss, TransformVariables

"""
    IsConstrained{T}
Trait to determine whether a proposal / model / ... `is_constrained(x)` (T=true) to a specific domain or not (T=false).
"""
struct IsConstrained{T} end

# Implementations for external types
function is_constrained(x::TransformVariables.TransformTuple)
  for t in x.transformations
    if !(typeof(t) <: TransformVariables.Identity)
      return IsConstrained{true}()
    end
  end
  IsConstrained{false}()
end

is_constrained(x::Soss.AbstractModel) = is_constrained(xform(x(;)))