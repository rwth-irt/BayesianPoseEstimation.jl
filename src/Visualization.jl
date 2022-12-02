# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using ColorSchemes
using CoordinateTransformations: SphericalFromCartesian
using Images
using IterTools: partition
using Plots
using Plots.PlotMeasures
using StatsBase
import StatsPlots: density as stats_density

const RWTH_blue = colorant"#00549f"
distinguishable_rwth(n) = distinguishable_colors(n, RWTH_blue, lchoices=0:75)

"""
    diss_default(;kwargs...)

"""
diss_defaults(; size=(148.4789, 83.5193), fontsize=11, fontfamily="helvetica", kwargs...) = Plots.default(; titlefontsize=correct_fontsize(fontsize), legendfontsize=correct_fontsize(fontsize), guidefontsize=correct_fontsize(fontsize), tickfontsize=correct_fontsize(0.8 * fontsize), colorbar_tickfontsize=correct_fontsize(0.8 * fontsize), annotationfontsize=correct_fontsize(fontsize), size=correct_size(size), fontfamily=fontfamily, colorbar_tickfontfamily=fontfamily, markersize=2, markerstrokewidth=0.5, kwargs...)

correct_fontsize(font_size) = correct_fontsize(Plots.backend(), font_size)
correct_fontsize(::Plots.AbstractBackend, font_size) = font_size |> round

correct_size(size) = correct_size(Plots.backend(), size)
# 4//3 plotly magic number??
correct_size(::Plots.AbstractBackend, size) = size .* (mm / px) .|> round

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
    plot = heatmap(transpose(img); colorbar_title=colorbar_title, color=color_grad, clims=clims, aspect_ratio=1, yflip=true, framestyle=:semi, xmirror=true, background_color_outside=:transparent, xlabel="x-pixels", ylabel="y-pixels", kwargs...)
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
function Base.convert(::Type{Matrix}, chain::AbstractVector{<:Sample}, var_name::Symbol, step=1)
    M = hcat([variables(chain[i])[var_name] for i in 1:step:length(chain)]...)
    # TODO should this be hidden in here?
    if M isa AbstractArray{<:Quaternion}
        M = map(M) do q
            r_q = QuatRotation(q)
            r_xyz = RotXYZ(r_q)
            [r_xyz.theta1, r_xyz.theta2, r_xyz.theta3]
        end
        M = hcat(M...)
    end
    M
end

"""
  convert(Matrix, chain, var_name::Symbol, step = 1)
Converts the chains to a column matrix of the variable `var_name`.
"""
Base.convert(::Type{Matrix}, chains::AbstractVector{<:AbstractVector{<:Sample}}, var_name::Symbol, step=1) = hcat([Base.convert(Matrix, c, var_name, step) for c in chains]...)

"""
  scatter_position(M; c_grad)
Creates a 3D scatter plot of the column matrix.
"""
function scatter_position(M::AbstractMatrix; c_grad=:viridis, kwargs...)
    # z value for color in order of the samples
    mz = [1:length(M[1, :])+1...]
    s = size(M)
    s = size(mz)
    scatter(M[1, :], M[2, :], M[3, :]; marker_z=mz, color=cgrad(c_grad), markersize=3, xlabel="x", ylabel="y", zlabel="z", kwargs...)
end

"""
    scatter_position(chains, [step]; c_grad, kwargs...)
Creates a 3D scatter plot of the chain for the given variable.
"""
scatter_position(chains::AbstractVector, step=1; c_grad=:viridis, kwargs...) = scatter_position(Base.convert(Matrix, chains, :t, step); c_grad=c_grad, label="sample number [÷$(step)]", kwargs...)

"""
    density_variable(chains, var_name, [step]; kwargs...)
Creates a density plot for the given variable.
"""
function histogram_variable(chains, var_name, step=1; kwargs...)
    M = convert(Matrix, chains, var_name, step)
    histogram(transpose(M); fill=true, fillalpha=0.4, kwargs...)
end

"""
    density_variable(chains, var_name, [step]; kwargs...)
Creates a density plot for the given variable.
"""
function density_variable(chains, var_name, step=1; kwargs...)
    M = convert(Matrix, chains, var_name, step)
    density(transpose(M); fill=true, fillalpha=0.4, palette=distinguishable_rwth(first(size(M))), trim=true, kwargs...)
end

"""
    polar_density_variable(chains, var_name, [step]; palette, kwargs...)
Creates a density plot in polar coordinates for the given variable.
"""
function polar_density_variable(chains, var_name, step=1; kwargs...)
    M = convert(Matrix, chains, var_name, step)
    stats_density(M', proj=:polar, fill=true, fillalpha=0.4, palette=distinguishable_rwth(first(size(M))), trim=true, kwargs...)
end

"""
    polar_histogram_variable(chains, var_name, [step], [nbins], palette)
Creates a histogram plot in polar coordinates for the given variable.
"""
function polar_histogram_variable(chains, var_name; step=1, nbins=90, kwargs...)
    M = convert(Matrix, chains, var_name, step)
    bins = range(0, 2π, length=nbins)
    pl = plot(palette=distinguishable_rwth(first(size(M))))
    for i in 1:size(M, 1)
        hist = fit(Histogram, M[i, :], bins)
        plot!(pl, bins[begin:end], append!(hist.weights, hist.weights[1]); proj=:polar, fill=true, fillalpha=0.4, kwargs...)
    end
    yticks!(pl, Float64[])
end

"""
    polar_variable(chains, var_name, [step]; kwargs....)
Line plot of the variable.
"""
function plot_variable(chains, var_name, step=1; kwargs...)
    M = convert(Matrix, chains, var_name, step)
    scatter(transpose(M); palette=distinguishable_rwth(first(size(M))), kwargs...)
end

"""
    plot_logprob(chains, [step]; kwargs...)
Plot of the logdensity over the samples.
"""
function plot_logprob(chains, step=1; kwargs...)
    logprobs = hcat([logprob(chains[i]) for i in 1:step:length(chains)]...)
    scatter(transpose(logprobs); palette=distinguishable_rwth(first(size(logprobs))), label="logprob", kwargs...)
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

"""
    sphere_density(rotations, [point]; [n_θ, n_ϕ, color], kwargs...)
Plot the density of the rotations by rotating a point on the unit sphere.
The density of the rotations is visualized as a heatmap and takes into account the non-uniformity of the patches' surface area on a sphere.
"""
function sphere_density(rotations, point=[0, 0, 1]; n_θ=50, n_ϕ=25, color=:viridis, kwargs...)
    # rotate on unit sphere
    r_points = map(r -> Vector(r * normalize(point)), rotations)
    # histogram of spherical coordinates: θ ∈ [-π,π], ϕ ∈ [-π/2,π/2]
    spherical = SphericalFromCartesian().(r_points)
    θ_hist = range(-π, π; length=n_θ + 1)
    ϕ_hist = range(-π / 2, π / 2; length=n_ϕ + 1)
    hist = fit(Histogram, (getproperty.(spherical, :θ), getproperty.(spherical, :ϕ)), (θ_hist, ϕ_hist))

    # sphere surface patches do not have a uniform area, calculate actual patch area using the sphere  integral for r=1
    ∫_unitsphere(θ_l, θ_u, ϕ_l, ϕ_u) = (cos(ϕ_l) - cos(ϕ_u)) * (θ_u - θ_l)
    # ∫_unitsphere(θ_l, θ_u, ϕ_l, ϕ_u) = (cos(θ_l) - cos(θ_u)) * (ϕ_u - ϕ_l)
    # different range than the spherical coordinates conversion above
    θ_patch = range(-π, π; length=n_θ + 1)
    ϕ_patch = range(0, π; length=n_ϕ + 1)
    patches = [∫_unitsphere(θ..., ϕ...) for θ in partition(θ_patch, 2, 1), ϕ in partition(ϕ_patch, 2, 1)]
    # area correction & max-norm
    weights = normalize(hist.weights ./ patches, Inf)
    # weights = hist.weights ./ patches

    # parametrize surface for the plot
    θ_surf = range(-π, π; length=n_θ)
    ϕ_surf = range(0, π; length=n_ϕ)
    x_surf = cos.(θ_surf) * sin.(ϕ_surf)'
    y_surf = sin.(θ_surf) * sin.(ϕ_surf)'
    z_surf = ones(n_θ) * cos.(ϕ_surf)'

    # override fill_z to use the weights for the surface color
    surface(x_surf, y_surf, z_surf; fill_z=weights, color=color, kwargs...)
end

"""
    sphere_density(rotations, [point]; [step, color, markersize, markeralpha], kwargs...)
Plot the density of the rotations by rotating a point on the unit sphere.
"""
function sphere_scatter(rotations, point=[0, 0, 1]; step=5, color=RWTH_blue, markersize=0.5, markeralpha=0.25, kwargs...)
    r_points = map(r -> Vector(r * point), rotations)
    r_scat = hcat(r_points...)
    scatter3d(r_scat[1, begin:step:end], r_scat[2, begin:step:end], r_scat[3, begin:step:end]; color=color, markersize=markersize, markeralpha=markeralpha, kwargs...)
end

"""
    sphere_density(rotations, [point]; [color, markersize, markeralpha], kwargs...)
Plot the density of the rotations by rotating a point on the unit sphere.
"""
function angleaxis_scatter(rotations; color=RWTH_blue, markersize=0.5, markeralpha=0.25, kwargs...)
    aa = AngleAxis.(rotations)
    x = getproperty.(aa, :axis_x) .* getproperty.(aa, :theta)
    y = getproperty.(aa, :axis_y) .* getproperty.(aa, :theta)
    z = getproperty.(aa, :axis_z) .* getproperty.(aa, :theta)
    scatter3d(x, y, z; color=color, markersize=markersize, markeralpha=markeralpha, kwargs...)
end
