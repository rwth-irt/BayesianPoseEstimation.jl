# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 
using BenchmarkTools
using MCMCDepth
using Test

# Tests
test_partial(a; x, y) = a + x + y
test_partial(1; x=2, y=3)
test_a = @inferred test_partial | (; x=2, y=3)
@inferred test_a(1)
@test test_a(1) == 6
test_x = @inferred test_partial | 1 | (; y=3)
@inferred test_x(; x=2)
@test test_x(; x=2) == 6
test_args = @inferred test_x | Val(:x)
@inferred test_args(2)
@test test_args(2) == 6

# BenchmarkTools
bench_x = @inferred test_partial | (; x=2)
bench_xy = @inferred bench_x | (; y=3)
@benchmark bench_xy(3)
@benchmark (test_partial | (; x=2, y=3))(2)
bench_ax = @inferred bench_x | (1,)
@benchmark bench_ax(; y=3)
bench_ax_s = @inferred test_partial | 1 | (; x=2)
@benchmark bench_ax(; y=3)
bench_kwargs_to_args = @inferred bench_ax | Val(:y)
# Faster than bench_ax? :D Nice!
@benchmark bench_kwargs_to_args(3)

# WARN it seems like having named parameters in the final function call is generally slower

# Test GPU execution
using CUDA
test_partial(a; x, y) = a + x - y
test_a = @inferred test_partial | (; x=2, y=3)
a = CUDA.fill(1, 100)
CUDA.@time test_a.(a)
# WARN cannot be conditioned on named parameter via broadcasting a CUDA array since NamedTuple broadcasting is reserved → move to args before broadcasting
test_kwargs_to_args = @inferred test_partial | Val(:y) | Val(:x)
@test test_kwargs_to_args(1, 2, 3) == 2
@test test_kwargs_to_args(1, 3, 2) == 0
broadcast_1 = broadcast(test_kwargs_to_args, a, 2, 3)
broadcast_2 = test_kwargs_to_args.(a, 2, 3)
b = Broadcast.broadcasted(test_kwargs_to_args, a, 2, 3)
broadcast_3 = Base.materialize(b)
@test broadcast_1 == broadcast_2 == broadcast_3
