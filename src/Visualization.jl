# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using ColorSchemes
using Images
using Plots
using StatsBase
using StatsPlots

"""
    normalize_depth(img)
Normalizes the values to [0,1].
"""
function normalize_depth(img)
    # OpenGL and Julia have different conventions for columns and rows
    depth = transpose(img[:, end:-1:1])
    # offset: ignore 0s by setting them to inf
    depth_img = [ifelse(iszero(x), Inf, x) for x in depth]
    depth_img = depth .- minimum(depth_img)
    # set inf to 0 again
    depth_img = [ifelse(isinf(x), zero(x), x) for x in depth_img]
    # scale
    depth_img = depth_img ./ maximum(depth_img)
end

"""
    plot_depth_img
Plot a depth image with a given `color_scheme` and use black for values of 0.
"""
function plot_depth_img(img; color_scheme=:viridis, reverse=true, colorbar_title="depth [m]")
    # Copy because of the inplace operations
    color_scheme = copy(color_list(color_scheme))
    if reverse
        reverse!(color_scheme)
    end
    pushfirst!(color_scheme, 0)
    min = minimum(img[img.>0])
    max = maximum(img)
    plot = heatmap(img; clims=(min, max), seriescolor=color_scheme, aspect_ratio=1, colorbar_title=colorbar_title)
    xlabel!(plot, "x-pixels")
    ylabel!(plot, "y-pixels")
end

"""
    plot_prob_img
Plot a probability image with a given `color_scheme` and use black for values of 0.
"""
plot_prob_img(img; color_scheme=:viridis, reverse=false, colorbar_title="probability [1]") = plot_depth_img(img; color_scheme=color_scheme, reverse=reverse, colorbar_title=colorbar_title)

"""
  convert(Matrix, chain, var_name::Symbol, step = 1)
Converts the chain to a column matrix of the variable `var_name`.
"""
Base.convert(::Type{Matrix}, chain::AbstractVector{<:Sample}, var_name::Symbol, step=1) = hcat([state(chain[i])[var_name] for i in 1:step:length(chain)]...)

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
function density_variable(chains, var_name, step=1, palette=:tol_bright)
    M = convert(Matrix, chains, var_name, step)
    Plots.density(transpose(M), fill=true, fillalpha=0.4, palette=palette)
end

"""
  polar_density_variable(chains, var_name, step, palette)
Creates a density plot in polar coordinates for the given variable.
"""
function polar_density_variable(chains, var_name, step=1, palette=:tol_bright)
    M = convert(Matrix, chains, var_name, step)
    StatsPlots.density(M', proj=:polar, fill=true, fillalpha=0.4, palette=palette)
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
  polar_histogram_variable(chains, var_name, step, nbins, palette)
Line plot of the variable.
"""
function plot_variable(chains, var_name, step=1, palette=:tol_bright, label=["x" "y" "z"])
    M = convert(Matrix, chains, var_name, step)
    scatter(transpose(M), palette=palette, markersize=2)
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
discrete_palette(cscheme=:viridis, length::Int64=3) = get(colorschemes[cscheme], range(0.0, 1.0, length=length))
