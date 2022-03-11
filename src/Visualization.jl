# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using ColorSchemes
using Images
using Plots
using StatsBase
using StatsPlots

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
end

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
end

"""
  convert(Matrix, chain, var_name::Symbol, step = 1)
Converts the chain to a column matrix of the variable `var_name`.
"""
Base.convert(::Type{Matrix}, chain::AbstractVector{<:Sample}, var_name::Symbol, step = 1) = hcat([state(chain[i])[var_name] for i in 1:step:length(chain)]...)

"""
  convert(Matrix, chain, var_name::Symbol, step = 1)
Converts the chains to a column matrix of the variable `var_name`.
"""
Base.convert(::Type{Matrix}, chains::AbstractVector{<:AbstractVector{<:Sample}}, var_name::Symbol, step = 1) = hcat([Base.convert(Matrix, c, var_name, step) for c in chains]...)

"""
  scatter_position(M, c_grad)
Creates a 3D scatter plot of the column matrix.
"""
function scatter_position(M::AbstractMatrix, c_grad = :viridis)
  # z value for color in order of the samples
  mz = [1:length(M[1, :])...]
  scatter(M[1, :], M[2, :], M[3, :], marker_z = mz, color = cgrad(c_grad), label = "Sample Number", markersize = 3, xlabel = "x", ylabel = "y", zlabel = "z")
end

"""
  scatter_position(chains, step, c_grad)
Creates a 3D scatter plot of the chain for the given variable.
"""
scatter_position(chains::AbstractVector, step = 1, c_grad = :viridis) = scatter_position(Base.convert(Matrix, chains, :t, step), c_grad)

"""
  density_variable(chains, var_name, step, palette)
Creates a density plot for the given variable.
"""
function density_variable(chains, var_name, step = 1, palette = :tol_bright)
  M = convert(Matrix, chains, var_name, step)
  Plots.density(transpose(M), fill = true, fillalpha = 0.4, palette = palette)
end

"""
  polar_density_variable(chains, var_name, step, palette)
Creates a density plot in polar coordinates for the given variable.
"""
function polar_density_variable(chains, var_name, step = 1, palette = :tol_bright)
  M = convert(Matrix, chains, var_name, step)
  StatsPlots.density(M', proj = :polar, fill = true, fillalpha = 0.4, palette = palette)
end

"""
  polar_histogram_variable(chains, var_name, step, nbins, palette)
Creates a histogram plot in polar coordinates for the given variable.
"""
function polar_histogram_variable(chains, var_name, step = 1, nbins = 90, palette = :tol_bright)
  M = convert(Matrix, chains, var_name, step)
  bins = range(0, 2π, length = nbins)
  pl = plot(palette = palette)
  for i in 1:size(M, 1)
    hist = fit(Histogram, M[i, :], bins)
    plot!(pl, bins[begin:end], append!(hist.weights, hist.weights[1]), proj = :polar, fill = true, fillalpha = 0.4)
  end
  yticks!(pl, Float64[])
end

"""
  polar_histogram_variable(chains, var_name, step, nbins, palette)
Line plot of the variable.
"""
function plot_variable(chains, var_name, step = 1, palette = :tol_bright, label = ["x" "y" "z"])
  M = convert(Matrix, chains, var_name, step)
  scatter(transpose(M), palette = palette, markersize = 2)
end

"""
  mean_prob_image(chain, var_name)
Creates an image of the mean of the given variable.
"""
mean_image(chain::AbstractVector{<:Sample}, var_name) = mean(x -> state(x)[var_name], chain)

"""
  mean_prob_image(chain, var_name)
Creates an image of the mean of the given variable.
"""
mean_image(chains::AbstractVector{<:AbstractVector{<:Sample}}, var_name) = mean(x -> mean_image(x, var_name), chains)

"""
  discrete_palette(cscheme, length)
Returns a discretized version of the color palette.
"""
discrete_palette(cscheme = :viridis, length::Int64 = 3) = get(colorschemes[cscheme], range(0.0, 1.0, length = length))
