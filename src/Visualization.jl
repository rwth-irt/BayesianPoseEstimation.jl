# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

import CairoMakie as MK
using ColorSchemes
using CoordinateTransformations: SphericalFromCartesian
using IterTools: partition

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
julia> fonts = PythonPlot.matplotlib.font_manager.FontManager().get_font_names()
```
"""

"""
Width of the document in pt
"""
const DISS_WIDTH = 422.52348

function diss_defaults()
    # GLMakie uses the original GLAbstractions, I hijacked GLAbstractions for my purposes
    MK.set_theme!(
        Axis=(; xticklabelsize=9, yticklabelsize=9, xgridstyle=:dash, ygridstyle=:dash),
        Axis3=(; xticklabelsize=9, yticklabelsize=9, zticklabelsize=9),
        CairoMakie=(; type="png", px_per_unit=2.0),
        Colorbar=(; width=7),
        Legend=(; patchsize=(5, 5), padding=(5, 5, 5, 5)),
        Lines=(; linewidth=1),
        Scatter=(; markersize=4),
        fontsize=11, # Latex "small" for normal 12
        resolution=(DISS_WIDTH, DISS_WIDTH / 2),
        rowgap=5, colgap=5,
        figure_padding=5
    )
end

# Image plotting

function plot_depth_img!(ax, img; colormap=:viridis, reverse=true, kwargs...)
    # Transfer to CPU
    img = Array(img)
    if reverse
        colormap = MK.Reverse(colormap)
    end
    min_depth = minimum(x -> x > 0 ? x : typemax(x), img)
    max_depth = maximum(x -> isinf(x) ? zero(x) : x, img)
    MK.heatmap!(ax, img; colormap=colormap, colorrange=(min_depth, max_depth), lowclip=:transparent, kwargs...)
end

function plot_prob_img!(ax, img; colormap=:viridis, reverse=false, kwargs...)
    # Transfer to CPU
    img = Array(img)
    # zero → black, one → white
    img = map(img) do x
        if x == 0
            typemin(x)
        elseif x == 1
            typemax(x)
        else
            x
        end
    end
    if reverse
        colormap = MK.Reverse(colormap)
    end
    MK.heatmap!(ax, img; colormap=colormap, colorrange=(0, 1), lowclip=:black, highclip=:white, kwargs...)
end

function img_fig_axis()
    fig = MK.Figure()
    ax = MK.Axis(fig[1, 1], xlabel="x-pixels", ylabel="y-pixels", aspect=1, yreversed=true)
    MK.hidedecorations!(ax; label=false, ticklabels=false, ticks=false)
    fig, ax
end

depth_hm_colorbar!(fig, depth_hm; label="depth / m") = MK.Colorbar(fig[:, end+1], depth_hm; label=label)

"""
    plot_prob_img
Plot a probability image with a given `color_scheme`.
Clips zero → black, one → white.
"""
function plot_prob_img(img; colorbar_label="probability [0,1]", kwargs...)
    fig, ax = img_fig_axis()
    prob_hm = plot_prob_img!(ax, img; kwargs...)
    depth_hm_colorbar!(fig, prob_hm; label=colorbar_label)
    fig
end

"""
    plot_depth_img(img; [value_to_typemax=0, reverse=true])
Plot a depth image with a given `color_scheme` and use black for values of 0.
`value_to_typemax` specifies the value which is converted to typemax.
`reverse` determines whether the color scheme is reversed.
"""
function plot_depth_img(img; colorbar_label="depth / m", kwargs...)
    fig, ax = img_fig_axis()
    depth_hm = plot_depth_img!(ax, img; kwargs...)
    depth_hm_colorbar!(fig, depth_hm; label=colorbar_label)
    fig
end

"""
    plot_depth_ontop(img, depth_img; [colorbar_label="depth / m", reverse=true, kwargs...])
Plot a depth image with a given `color_scheme` on top of another image.
`reverse` determines whether the color scheme is reversed.

See also [`plot_scene_ontop`](@ref), [`plot_best_pose`](@ref).
"""
function plot_depth_ontop(img, depth_img; colorbar_label="depth / m", kwargs...)
    fig, ax = img_fig_axis()
    # Plot the image as background
    MK.image!(ax, img; aspect=1)
    depth_hm = plot_depth_img!(ax, depth_img; alpha=0.7, kwargs...)
    depth_hm_colorbar!(fig, depth_hm; label=colorbar_label)
    fig
end

"""
    plot_scene_ontop(gl_context, scene, img; kwargs...)
Plot a scene as depth image on top of another image.

See also [`plot_depth_ontop`](@ref), [`plot_best_pose`](@ref).
"""
function plot_scene_ontop(gl_context, scene, img; kwargs...)
    render_img = draw(gl_context, scene)
    plot_depth_ontop(img, render_img; kwargs...)
end

"""
    plot_best_pose(chain, experiment, img; [getter=loglikelihood, kwargs...])
Plot the best pose of a chain/sample as depth image on top of another image.
Use the `getter` parameter to specify whether `loglikelihood` or `logprobability` should be used.

See also [`plot_depth_ontop`](@ref), [`plot_scene_ontop`](@ref).
"""
function plot_best_pose(sample::Sample, experiment, img, getter=loglikelihood; kwargs...)
    _, ind = findmax(getter(sample))
    scene = experiment.scene
    mesh = first(scene.meshes)
    @reset mesh.pose = to_pose(variables(sample).t[:, ind], variables(sample).r[ind])
    @reset scene.meshes = [mesh]
    plot_scene_ontop(experiment.gl_context, scene, img; kwargs...)
end

function plot_best_pose(chain::AbstractVector{<:Sample}, experiment, img, getter=loglikelihood; kwargs...)
    _, ind = findmax((s) -> getter(s), chain)
    scene = experiment.scene
    mesh = first(scene.meshes)
    @reset mesh.pose = to_pose(chain[ind].variables.t, chain[ind].variables.r)
    @reset scene.meshes = [mesh]
    plot_scene_ontop(experiment.gl_context, scene, img; kwargs...)
end

"""
    mean_image(sample, var_name)
Creates an image of the mean of the given variable.
"""
mean_image(sample::Sample, var_name) = dropdims(mean(variables(sample)[var_name], dims=3); dims=3)
mean_image(chain::AbstractVector{<:Sample}, var_name) = mean(x -> variables(x)[var_name], chain)
mean_image(chains::AbstractVector{<:AbstractVector{<:Sample}}, var_name) = mean(x -> mean_image(x, var_name), chains)

# Position and orientation conversion

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

# Densities

"""
    density_variable(chain, var_name; kwargs...)
Creates a density plot for the given variable.
"""
function density_variable!(axis, chain, var_name; palette=MK.Makie.wong_colors(), labels=nothing, kwargs...)
    M = plotable_matrix(chain, var_name)
    for (idx, values) in enumerate(eachrow(M))
        lbl = isnothing(labels) ? nothing : labels[idx]
        MK.density!(axis, values; label=lbl, strokewidth=1, color=(palette[idx], 0.45), strokecolor=palette[idx])
    end
end

"""
    scatter_variable(chain, var_name, [len=100]; kwargs....)
Line plot of the variable.
"""
function scatter_variable!(axis, chain, var_name, len=100; palette=MK.Makie.wong_colors(), labels=nothing, kwargs...)
    M = plotable_matrix(chain, var_name, len)
    for (idx, values) in enumerate(eachrow(M))
        lbl = isnothing(labels) ? nothing : labels[idx]
        MK.scatter!(axis, values; label=lbl, kwargs...)
    end
end

function plot_pose_density(sample; kwargs...)
    fig = MK.Figure(; resolution=(DISS_WIDTH, 1 / 3 * DISS_WIDTH))
    ax_t = MK.Axis(fig[1, 1]; xlabel="position / m", ylabel="density")
    density_variable!(ax_t, sample, :t; labels=["x" "y" "z"], kwargs...)
    ax_r = MK.Axis(fig[1, 2]; xlabel="orientation / rad", ylabel="density")
    density_variable!(ax_r, sample, :r; labels=["x" "y" "z"], kwargs...)
    MK.axislegend(ax_t; position=:ct)
    fig
end


plot_pose_density(state::SmcState; kwargs...) = plot_pose_density(state.sample; weights=exp.(state.log_weights), kwargs...)

function plot_pose_chain(model_chain, len=50)
    fig = MK.Figure(resolution=(DISS_WIDTH, 2 / 3 * DISS_WIDTH))
    ax_ts = MK.Axis(fig[1, 1]; xlabel="iteration ÷ $(length(model_chain) ÷ len)", ylabel="position / m")
    scatter_variable!(ax_ts, model_chain, :t, len; labels=["x" "y" "z"])
    ax_td = MK.Axis(fig[2, 1]; xlabel="position / m", ylabel="density")
    density_variable!(ax_td, model_chain, :t; labels=["x" "y" "z"])

    ax_rs = MK.Axis(fig[1, 2]; xlabel="iteration ÷ $(length(model_chain) ÷ len)", ylabel="orientation / rad")
    scatter_variable!(ax_rs, model_chain, :r, len; labels=["x" "y" "z"])
    ax_rd = MK.Axis(fig[2, 2]; xlabel="orientation / rad", ylabel="density")
    density_variable!(ax_rd, model_chain, :r; labels=["x" "y" "z"])
    MK.axislegend(ax_td; position=:ct)
    fig
end

plot_logprob(logprobs::AbstractVector{<:Number}; xlabel="iteration", ylabel="log probability", kwargs...) = MK.scatter(logprobs; axis=(; xlabel=xlabel, ylabel=ylabel), kwargs...)

"""
    plot_logprob(chains, [len=100]; kwargs...)
Plot of the logdensity over the samples.
"""
plot_logprob(chain::AbstractVector{<:Sample}, len=100; kwargs...) = plot_logprob(step_data(logprobability.(chain)); xlabel="iteration ÷ $(length(chain) ÷ len)")
plot_logprob(final_sample::Sample, len=100; kwargs...) = plot_logprob(step_data(logprobability(final_sample)); xlabel="iteration ÷ $(length(chain) ÷ len)")

"""
    plot_logevidence(states, [len=100]; kwargs...)
Plot of the logdensity of the SMC states.
"""
plot_logevidence(chain::AbstractVector{<:SmcState}, len=100; kwargs...) = plot_logprob(step_data(logevidence.(chain), len); ylabel="log evidence", xlabel="iteration ÷ $(length(chain) ÷ len)")
