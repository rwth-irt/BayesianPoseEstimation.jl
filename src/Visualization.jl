# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using ColorSchemes
using CoordinateTransformations: SphericalFromCartesian
using IterTools: partition
using Plots
using Plots.PlotMeasures
import StatsPlots: density as stats_density

const RWTH_blue = colorant"#00549f"
distinguishable_rwth(n) = distinguishable_colors(n, RWTH_blue, lchoices=0:75)

"""
    diss_default(;kwargs...)
Install fonts for matplotlib
    - Arial = Helvetica
    - Times New Romans
    - Carlito (google) = Calibri
    - Caladea (google) = Cambria
```bash
apt install ttf-mscorefonts-installer fonts-crosextra-carlito fonts-crosextra-caladea
fc-cache
rm -rf ~/.cache/matplotlib
julia> fonts = PyPlot.matplotlib.font_manager.FontManager().get_font_names()
```
"""
diss_defaults(; size=(148.4789, 83.5193), fontsize=11, fontfamily="Helvetica", kwargs...) = Plots.default(; titlefontsize=correct_fontsize(fontsize), legendfontsize=correct_fontsize(fontsize), guidefontsize=correct_fontsize(fontsize), tickfontsize=correct_fontsize(0.8 * fontsize), colorbar_tickfontsize=correct_fontsize(0.8 * fontsize), annotationfontsize=correct_fontsize(fontsize), size=correct_size(size), fontfamily=fontfamily, colorbar_tickfontfamily=fontfamily, markersize=2, markerstrokewidth=0.5, kwargs...)

correct_fontsize(font_size) = correct_fontsize(Plots.backend(), font_size)
correct_fontsize(::Plots.AbstractBackend, font_size) = font_size |> round

correct_size(size) = correct_size(Plots.backend(), size)
# 4//3 plotly magic number??
correct_size(::Plots.AbstractBackend, size) = size .* (mm / px) .|> round

# Image plotting

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
function plot_depth_img(img; value_to_typemax=0, color_scheme=:viridis, reverse=true, colorbar_title="depth / m", clims=nothing, kwargs...)
    # Transfer to CPU
    img = Array(img)
    # color_grad = cgrad(color_scheme; rev=reverse)
    color_grad = cgrad(color_scheme; rev=reverse)
    # pushfirst!(color_scheme, 0)
    mask = img .!= value_to_typemax
    if clims === nothing
        min = minimum(img[mask])
        max = maximum(img)
        clims = (min, max)
    end
    img = value_or_typemax.(img, value_to_typemax)
    # GR has some overlap of the colorbar_title
    if Plots.backend() == Plots.GRBackend()
        colorbar_title = " \n" * colorbar_title
        kwargs = (; kwargs, right_margin=8Plots.pt)
    end
    heatmap(transpose(img); colorbar_title=colorbar_title, color=color_grad, clims=clims, aspect_ratio=1, yflip=true, framestyle=:semi, xmirror=true, background_color_outside=:transparent, xlabel="x-pixels", ylabel="y-pixels", kwargs...)
end

"""
    plot_depth_img(img, depth_img; [value_to_typemax=0, reverse=true])
Plot a depth image with a given `color_scheme` on top of another image.
`value_to_typemax` specifies the value which is converted to typemax.
`reverse` determines whether the color scheme is reversed.
"""
function plot_depth_ontop(img, depth_img; value_to_typemax=0, color_scheme=:viridis, reverse=true, colorbar_title="depth / m", clims=nothing, alpha=0.5, kwargs...)
    # Plot the image as background
    plot(img)
    # Transfer to CPU
    depth_img = Array(depth_img)
    # color_grad = cgrad(color_scheme; rev=reverse)
    color_grad = cgrad(color_scheme; rev=reverse)
    # pushfirst!(color_scheme, 0)
    mask = depth_img .!= value_to_typemax
    if clims === nothing
        min = minimum(depth_img[mask])
        max = maximum(depth_img)
        clims = (min, max)
    end
    depth_img = value_or_typemax.(depth_img, value_to_typemax)
    # GR has some overlap of the colorbar_title
    if Plots.backend() == Plots.GRBackend()
        colorbar_title = " \n" * colorbar_title
        kwargs = (; kwargs, right_margin=8Plots.pt)
    end
    # Plot the depth image on top
    heatmap!(transpose(depth_img); alpha=alpha, colorbar_title=colorbar_title, color=color_grad, clims=clims, aspect_ratio=1, yflip=true, framestyle=:semi, xmirror=true, background_color_outside=:transparent, xlabel="x-pixels", ylabel="y-pixels", kwargs...)
end

"""
    plot_prob_img
Plot a probability image with a given `color_scheme` and use black for values of 0.
"""
plot_prob_img(img; color_scheme=:viridis, reverse=false, colorbar_title="probability [0,1]", kwargs...) = plot_depth_img(img; value_to_typemax=nothing, color_scheme=color_scheme, reverse=reverse, colorbar_title=colorbar_title, clims=(0, 1), kwargs...)

"""
    mean_image(chain, var_name)
Creates an image of the mean of the given variable.
"""
mean_image(chain::AbstractVector{<:Sample}, var_name) = mean(x -> variables(x)[var_name], chain)

"""
    mean_image(chain, var_name)
Creates an image of the mean of the given variable.
"""
mean_image(chains::AbstractVector{<:AbstractVector{<:Sample}}, var_name) = mean(x -> mean_image(x, var_name), chains)

# Position and Orientation conversion

"""
    step_data(A, len)
Returns a view of the sub data where the last dimensions has length `len`.
"""
step_data(A, len) = @view A[.., round.(Int, LinRange(1, last(size(A)), len))]

plotable_matrix(vec::AbstractVector) = hcat(vec...)
plotable_matrix(M::AbstractMatrix) = M

plotable_matrix(vec::AbstractVector{<:Quaternion}) = plotable_matrix(rotxyz_vector.(vec))

function rotxyz_vector(q::Quaternion)
    r_q = QuatRotation(q)
    r_xyz = RotXYZ(r_q)
    [r_xyz.theta1, r_xyz.theta2, r_xyz.theta3]
end

"""
    position_matrix(chain, var_name, [len=n_samples])
Generate a Matrix with [x,y,z;len] for each position in the chain.
"""
plotable_matrix(chain::AbstractVector{<:Sample}, var_name, len=length(chain)) = plotable_matrix(getindex.(variables.(step_data(chain, len)), [var_name]))

plotable_matrix(final_sample::Sample, var_name, len=last(size(variables(final_sample)[var_name]))) = plotable_matrix(step_data(variables(final_sample)[var_name], len))

# Pose plotting

function plot_pose_density(sample; kwargs...)
    plt_t_dens = density_variable(sample, :t; label=["x" "y" "z"], xlabel="Position / m", ylabel="Density", legend=false, kwargs...)

    plt_r_dens = density_variable(sample, :r; label=["x" "y" "z"], xlabel="Orientation / rad", ylabel="Density", legend=false, kwargs...)

    plot(
        plt_t_dens, plt_r_dens,
        layout=(2, 1)
    )
end

function plot_pose_chain(model_chain, len=50)
    plt_t_chain = plot_variable(model_chain, :t, len; label=["x" "y" "z"], xlabel="Iteration [÷ $(len)]", ylabel="Position / m", legend=false)
    plt_t_dens = density_variable(model_chain, :t; label=["x" "y" "z"], xlabel="Position / m", ylabel="Density", legend=false, left_margin=5mm)

    plt_r_chain = plot_variable(model_chain, :r, len; label=["x" "y" "z"], xlabel="Iteration [÷ $(len)]", ylabel="Orientation / rad", legend=false, top_margin=5mm)
    plt_r_dens = density_variable(model_chain, :r; label=["x" "y" "z"], xlabel="Orientation / rad", ylabel="Density", legend=false)

    plot(
        plt_t_chain, plt_r_chain,
        plt_t_dens, plt_r_dens,
        layout=(2, 2)
    )
end

"""
    scatter_position(M; c_grad)
Creates a 3D scatter plot of the matrix.
"""
function scatter_position(M::AbstractMatrix; c_grad=:viridis, kwargs...)
    # z value for color in order of the samples
    mz = [1:length(M[1, :])+1...]
    s = size(M)
    s = size(mz)
    scatter(M[1, :], M[2, :], M[3, :]; marker_z=mz, color=cgrad(c_grad), markersize=3, xlabel="x", ylabel="y", zlabel="z", kwargs...)
end

function scatter_position(chain, len=100; var_name=:t, c_grad=:viridis, kwargs...)
    M = plotable_matrix(chain, var_name, len)
    step = round(last(size(M)) / len; digits=1)
    scatter_position(M; c_grad=c_grad, label="sample number [÷$(step)]", kwargs...)
end

"""
    density_variable(chain, var_name; kwargs...)
Creates a density plot for the given variable.
"""
function histogram_variable(chain, var_name; kwargs...)
    M = plotable_matrix(chain, var_name)
    histogram(transpose(M); fill=true, fillalpha=0.4, kwargs...)
end

"""
    density_variable(chain, var_name; kwargs...)
Creates a density plot for the given variable.
"""
function density_variable(chain, var_name; kwargs...)
    M = plotable_matrix(chain, var_name)
    density(transpose(M); fill=true, fillalpha=0.4, palette=distinguishable_rwth(first(size(M))), trim=true, kwargs...)
end

"""
    polar_density_variable(chain, var_name; kwargs...)
Creates a density plot in polar coordinates for the given variable.
"""
function polar_density_variable(chain, var_name; kwargs...)
    M = plotable_matrix(chain, var_name)
    density(M', proj=:polar, fill=true, fillalpha=0.4, palette=distinguishable_rwth(first(size(M))), trim=true, kwargs...)
end

"""
    polar_histogram_variable(chain, var_name, [nbins], palette)
Creates a histogram plot in polar coordinates for the given variable.
"""
function polar_histogram_variable(chain, var_name; nbins=90, kwargs...)
    M = plotable_matrix(chain, var_name)
    bins = range(0, 2π, length=nbins)
    pl = plot(palette=distinguishable_rwth(first(size(M))))
    for i in 1:size(M, 1)
        hist = fit(Histogram, M[i, :], bins)
        plot!(pl, bins[begin:end], append!(hist.weights, hist.weights[1]); proj=:polar, fill=true, fillalpha=0.4, kwargs...)
    end
    yticks!(pl, Float64[])
end

"""
    plot_variable(chain, var_name, [len=100]; kwargs....)
Line plot of the variable.
"""
function plot_variable(chain, var_name, len=100; kwargs...)
    M = plotable_matrix(chain, var_name, len)
    scatter(transpose(M); palette=distinguishable_rwth(first(size(M))), kwargs...)
end

plot_logprob(logprobs::AbstractVector{<:Number}; kwargs...) = scatter(logprobs; palette=distinguishable_rwth(first(size(logprobs))), label="logprob", kwargs...)

"""
    plot_logprob(chains, [len=100]; kwargs...)
Plot of the logdensity over the samples.
"""
plot_logprob(chain::AbstractVector{<:Sample}, len=100; kwargs...) = plot_logprob(step_data(logprob.(chain), len))
plot_logprob(final_sample::Sample, len=100; kwargs...) = plot_logprob(step_data(logprob(final_sample), len))

"""
    discrete_palette(cscheme, [len=3])
Returns a discretized version of the color palette.
"""
discrete_palette(cscheme=:viridis, len::Int64=3) = get(colorschemes[cscheme], range(0.0, 1.0, length=len))

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

function sphere_density(chain::AbstractVector{<:Sample}, var_name=:r, point=[0, 0, 1]; kwargs...)
    rotations = QuatRotation.(getindex.(variables.(step_data(chain, length(chain))), var_name))
    sphere_density(rotations, point; kwargs...)
end

function sphere_density(final_sample::Sample, var_name=:r, point=[0, 0, 1]; kwargs...)
    rotations = QuatRotation.(variables(final_sample)[var_name])
    sphere_density(rotations, point; kwargs...)
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
