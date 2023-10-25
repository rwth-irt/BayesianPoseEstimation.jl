# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
import CairoMakie as MK
using MCMCDepth
using PoseErrors
using Random
using SciGL

parameters = Parameters()
@reset parameters.device = :CPU
@reset parameters.width = 150
@reset parameters.height = 150

# Takes minutes instead of seconds
# @reset parameters.device = :CPU
rng = device_rng(parameters)
gl_context = render_context(parameters)
df = gt_targets(joinpath("data", "bop", "lmo", "test"), 2)
row = df[14, :]
camera = crop_camera(row)
mesh = upload_mesh(gl_context, load_mesh(row))
@reset mesh.pose = to_pose(row.gt_t, row.gt_R)
scene = Scene(camera, [mesh])

# Color image
color_img = load_color_image(row, parameters.img_size...)
# Measured image
depth_img = load_depth_image(row, parameters.img_size...) |> device_array_type(parameters)
# Expected / perfect image
render_img = draw(gl_context, scene)

# Load data for probabilistic model
mask_img = load_mask_image(row, parameters.img_size...) |> device_array_type(parameters)
prior_t = point_from_segmentation(row.bbox, depth_img, mask_img, row.cv_camera)
prior_o = fill(parameters.float_type(0.5), parameters.width, parameters.height)

# Probabilistic model   
t = BroadcastedNode(:t, rng, KernelNormal, prior_t, parameters.σ_t)
r = BroadcastedNode(:r, rng, QuaternionUniform, parameters.float_type)
μ_fn = render_fn | (gl_context, scene)
μ = DeterministicNode(:μ, μ_fn, (t, r))
o = BroadcastedNode(:o, rng, KernelDirac, prior_o)
z_i = pixel_mixture | (parameters.min_depth, parameters.max_depth, parameters.pixel_θ, parameters.pixel_σ)
z = BroadcastedNode(:z, rng, z_i, (μ, o))
Random.seed!(rng, 8)
s = rand(z)
gen_img = copy(s.z)

# With mask
prior_o = fill(parameters.float_type(0.2), parameters.width, parameters.height) |> device_array_type(parameters)
prior_o[mask_img] .= 0.8
o = BroadcastedNode(:o, rng, KernelDirac, prior_o)
z_i = pixel_mixture | (parameters.min_depth, parameters.max_depth, parameters.pixel_θ, parameters.pixel_σ)
z = BroadcastedNode(:z, rng, z_i, (μ, o))
Random.seed!(rng, 8)
s = rand(z)
masked_img = copy(s.z)

destroy_context(gl_context)

# Compose figures
diss_defaults()

# Simple generative model
fig = MK.Figure(resolution=(DISS_WIDTH, 1 / 3 * DISS_WIDTH))
grid_meas = MK.GridLayout(fig[1, 1])
ax_exp = img_axis(fig[1, 2]; title="expectation μ", ylabel="")
ax_gen = img_axis(fig[1, 3]; title="μ + noise", ylabel="")
plot_depth_ontop!(grid_meas, color_img, depth_img; title="measurement z")
hm = plot_depth_img!(ax_exp, render_img)
plot_depth_img!(ax_gen, gen_img)
cb = MCMCDepth.heatmap_colorbar!(fig, hm; label="", ticks=([minimum(render_img[render_img.>0]) + 0.01, maximum(render_img) - 0.01], ["close", "far"]))
display(fig)
MK.save(joinpath("plots", "gen_depth.pdf"), fig)

# Using the prior
fig = MK.Figure(resolution=(DISS_WIDTH, 1 / 3 * DISS_WIDTH))
grid_meas = MK.GridLayout(fig[1, 1])
ax_unin = img_axis(fig[1, 2]; title="uninformed prior", ylabel="")
ax_mask = img_axis(fig[1, 3]; title="mask prior", ylabel="")
plot_depth_ontop!(grid_meas, color_img, depth_img; title="measurement z")
plot_depth_img!(ax_unin, gen_img)
hm = plot_depth_img!(ax_mask, masked_img)
cb = MCMCDepth.heatmap_colorbar!(fig, hm; label="", ticks=([minimum(masked_img[masked_img.>0]) + 0.01, maximum(masked_img) - 0.01], ["close", "far"]))
display(fig)
MK.save(joinpath("plots", "gen_depth_prior_o.pdf"), fig)
