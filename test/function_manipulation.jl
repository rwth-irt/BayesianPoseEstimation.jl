# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 
using BenchmarkTools
using MCMCDepth

# Tests
test_partial(a; x, y) = a + x + y
test_partial(1; x=2, y=3)
test_a = test_partial | (; x=2, y=3)
test_a(1)
@code_warntype test_a(1)
test_x = test_partial | 1 | (; y=3)
test_x(; x=2)
@code_warntype test_x(; x=2)
test_args = test_x | :x
test_args(2)
@code_warntype test_args(2)

# BenchmarkTools
bench_x = test_partial | (; x=2)
bench_xy = bench_x | (; y=3)
@benchmark bench_xy(3)
@benchmark (test_partial | (; x=2, y=3))(2)
bench_ax = bench_x | (1,)
@benchmark bench_ax(; y=3)
bench_ax_s = test_partial | 1 | (; x=2)
@benchmark bench_ax(; y=3)
bench_kwargs_to_args = bench_ax | :y
# Faster than bench_ax? :D Nice!
@benchmark bench_kwargs_to_args(3)