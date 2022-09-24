# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using ColorSchemes
using Images
using Plots
using Plots.PlotMeasures
using StatsBase
import StatsPlots: density as stats_density

"""
   value_or_typemax(x, [value=zero(x)])
If the x != value, x is returned otherwise typemax.
Use it to generate the background for the heatmap.
"""
value_or_typemax(x, value=zero(x)) = x != value ? x : typemax(x)

"""
    plot_depth_img(img; [value_to_typemax=0, reverse=true])
Plot a depth image with a given `color_scheme` and use black for values of 0.
`value_to_typemax` specifies the value which is converted to typemax.
`reverse` determines whether the color scheme is reversed.
"""
function plot_depth_img(img; value_to_typemax=0, color_scheme=:viridis, reverse=true, colorbar_title="depth [m]", clims=nothing, kwargs...)
  # Copy because of the inplace operations
  # color_grad = cgrad(color_scheme; rev=reverse)
  color_grad = cgrad(color_scheme; rev=reverse)
  # pushfirst!(color_scheme, 0)
  mask = img .!= value_to_typemax
  if clims === nothing
    min = minimum(img[mask])
    max = maximum(img)
    clims = (min, max)
  end
  width, height = size(img)
  img = value_or_typemax.(img, value_to_typemax)
  plot = heatmap(transpose(img); colorbar_title=colorbar_title, color=color_grad, clims=clims, aspect_ratio=1, yflip=true, framestyle=:zerolines, x_ticks=[width], xmirror=true, y_ticks=[0, height], background_color_outside=:transparent, kwargs...)

  xlabel!(plot, "x-pixels")
  ylabel!(plot, "y-pixels")
end

"""
    plot_prob_img
Plot a probability image with a given `color_scheme` and use black for values of 0.
"""
plot_prob_img(img; color_scheme=:viridis, reverse=false, colorbar_title="probability [0,1]", kwargs...) = plot_depth_img(img; value_to_typemax=nothing, color_scheme=color_scheme, reverse=reverse, colorbar_title=colorbar_title, clims=(0, 1), kwargs...)

"""
  convert(Matrix, chain, var_name::Symbol, step = 1)
Converts the chain to a column matrix of the variable `var_name`.
"""
Base.convert(::Type{Matrix}, chain::AbstractVector{<:Sample}, var_name::Symbol, step=1) = hcat([variables(chain[i])[var_name] for i in 1:step:length(chain)]...)

"""
  convert(Matrix, chain, var_name::Symbol, step = 1)
Converts the chains to a column matrix of the variable `var_name`.
"""
Base.convert(::Type{Matrix}, chains::AbstractVector{<:AbstractVector{<:Sample}}, var_name::Symbol, step=1) = hcat([Base.convert(Matrix, c, var_name, step) for c in chains]...)

"""
  scatter_position(M, c_grad)
Creates a 3D scatter plot of the column matrix.
"""
function scatter_position(M::AbstractMatrix, c_grad=:viridis)
  # z value for color in order of the samples
  mz = [1:length(M[1, :])...]
  scatter(M[1, :], M[2, :], M[3, :], marker_z=mz, color=cgrad(c_grad), label="Sample Number", markersize=3, xlabel="x", ylabel="y", zlabel="z")
end

"""
  scatter_position(chains, step, c_grad)
Creates a 3D scatter plot of the chain for the given variable.
"""
scatter_position(chains::AbstractVector, step=1, c_grad=:viridis) = scatter_position(Base.convert(Matrix, chains, :t, step), c_grad)

"""
  density_variable(chains, var_name, step, palette)
Creates a density plot for the given variable.
"""
function histogram_variable(chains, var_name, step=1, palette=:tol_bright)
  M = convert(Matrix, chains, var_name, step)
  histogram(transpose(M), fill=true, fillalpha=0.4, palette=palette)
end

"""
  density_variable(chains, var_name, step, palette)
Creates a density plot for the given variable.
"""
function density_variable(chains, var_name, step=1, palette=:tol_bright)
  M = convert(Matrix, chains, var_name, step)
  density(transpose(M), fill=true, fillalpha=0.4, palette=palette, trim=true)
end

"""
  polar_density_variable(chains, var_name, step, palette)
Creates a density plot in polar coordinates for the given variable.
"""
function polar_density_variable(chains, var_name, step=1, palette=:tol_bright)
  M = convert(Matrix, chains, var_name, step)
  stats_density(M', proj=:polar, fill=true, fillalpha=0.4, palette=palette, trim=true)
end

"""
  polar_histogram_variable(chains, var_name, step, nbins, palette)
Creates a histogram plot in polar coordinates for the given variable.
"""
function polar_histogram_variable(chains, var_name, step=1, nbins=90, palette=:tol_bright)
  M = convert(Matrix, chains, var_name, step)
  bins = range(0, 2π, length=nbins)
  pl = plot(palette=palette)
  for i in 1:size(M, 1)
    hist = fit(Histogram, M[i, :], bins)
    plot!(pl, bins[begin:end], append!(hist.weights, hist.weights[1]), proj=:polar, fill=true, fillalpha=0.4)
  end
  yticks!(pl, Float64[])
end

"""
  polar_variable(chains, var_name, step, nbins, palette)
Line plot of the variable.
"""
function plot_variable(chains, var_name, step=1, palette=:tol_bright, label=["x" "y" "z"])
  M = convert(Matrix, chains, var_name, step)
  scatter(transpose(M), palette=palette, markersize=2)
end

"""
  plot_logprob(chains, var_name, step, nbins, palette)
Plot of the logdensity over the samples.
"""
function plot_logprob(chains, step=1, palette=:tol_bright)
  logprobs = hcat([logprob(chains[i]) for i in 1:step:length(chains)]...)
  scatter(transpose(logprobs); palette=palette, markersize=2, label="logprob")
end

"""
  mean_prob_image(chain, var_name)
Creates an image of the mean of the given variable.
"""
mean_image(chain::AbstractVector{<:Sample}, var_name) = mean(x -> variables(x)[var_name], chain)

"""
  mean_prob_image(chain, var_name)
Creates an image of the mean of the given variable.
"""
mean_image(chains::AbstractVector{<:AbstractVector{<:Sample}}, var_name) = mean(x -> mean_image(x, var_name), chains)

"""
  discrete_palette(cscheme, length)
Returns a discretized version of the color palette.
"""
discrete_palette(cscheme=:viridis, length::Int64=3) = get(colorschemes[cscheme], range(0.0, 1.0, length=length))
