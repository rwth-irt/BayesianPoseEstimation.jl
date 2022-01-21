# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using ColorSchemes
using Images

"""
    colorize_depth(img; color_scheme, rev)
Takes a depth image `img` which is some kind of Matrix{Float}, normalizes the values to [0,1] and colorizes it using the `color_scheme`.
"""
function colorize_depth(img; color_scheme = :viridis, rev = true)
    # OpenGL and Julia have different conventions for columns and rows
    depth = transpose(img[:, end:-1:1])
    # offset: ignore 0s by setting them to inf
    depth_img = [ifelse(iszero(x), Inf, x) for x in depth]
    depth_img = depth .- minimum(depth_img)
    # set inf to 0 again
    depth_img = [ifelse(isinf(x), zero(x), x) for x in depth_img]
    # scale
    depth_img = depth_img ./ maximum(depth_img)
    # colorize
    c_scheme = ColorSchemes.eval(color_scheme)
    if rev
        c_scheme = reverse(c_scheme)
    end
    # Colorize only foreground
    [ifelse(x > 0, c_scheme[x], RGB()) for x in depth_img]
end;

"""
    colorize_depth(img; color_scheme, rev)
Takes a probability image `img` which has values ∈ [0,1], `color_scheme`.
"""
function colorize_probability(img; color_scheme = :viridis, rev = false)
    # OpenGL and Julia have different conventions for columns and rows
    prob_img = transpose(img[:, end:-1:1])
    # colorize
    c_scheme = ColorSchemes.eval(color_scheme)
    if rev
        c_scheme = reverse(c_scheme)
    end
    # Colorize only foreground
    [ifelse(x > 0, c_scheme[x], RGB()) for x in prob_img]
end;
