using Accessors
using CUDA
using Distributions
using LinearAlgebra
using MCMCDepth
using Plots
using Random
using SciGL
using Test

rng = Random.default_rng()
Random.seed!(rng, 42)

# Setup render context & scene
params = MCMCDepth.Parameters()
params = @set params.mesh_files = ["meshes/BM067R.obj"]
render_context = RenderContext(params.width, params.height, params.depth, Array)

# Setup probability Distributions
function mix_normal_truncated_exponential(σ::T, θ::T, μ::T, o::T) where {T<:Real}
    # TODO should these generators be part of experiment specific scripts or should I provide some default ones?
    # TODO Compare whether truncated even makes a difference
    dist = KernelBinaryMixture(KernelNormal(μ, σ), truncated(KernelExponential(θ), nothing, μ), o, one(o) - o)
    ValidPixel(μ, dist)
end
my_pixel_dist = mix_normal_truncated_exponential | (0.1f0, 1.0f0)

# CvCamera like ROS looks down positive z
scene = Scene(params, render_context)
t = [-0.05, 0.05, 0.25]
r = [1, 1, 0]
p = to_pose(t, r)
μ = copy(render(render_context, scene, 1, p))
plot_depth_img(μ)

obs_params = @set params.mesh_files = ["meshes/BM067R.obj", "meshes/cube.obj"]
obs_scene = Scene(obs_params, render_context)
obs_scene = @set obs_scene.meshes[2].pose.translation = [-0.5, 0.5, 0.6]
obs_t = [-0.05, 0.048, 0.25]
obs_r = [1, 1.03, 0]
obs_p = to_pose(obs_t, obs_r)
obs_μ = copy(render(render_context, obs_scene, 1, obs_p))
plot_depth_img(obs_μ)

μ2 = copy(render(render_context, scene, 1, obs_p))
plot_depth_img(μ - μ2; colorbar_title="depth difference [m]")

o = rand(rng, KernelUniform(0.9f0, 1.0f0), 100, 100)
img = rand(my_pixel_dist.(obs_μ, o))

# install fonts for matplotlib (Arial, Times New Romans & Calibri / Cambria replacements)
#$ apt install ttf-mscorefonts-installer fonts-crosextra-carlito fonts-crosextra-caladea
#$ fc-cache
#$ rm -rf ~/.cache/matplotlib
#julia> fonts = PyPlot.matplotlib.font_manager.FontManager().get_font_names()
pyplot()
MCMCDepth.diss_defaults(; fontfamily="Carlito", fontsize=11, size=(160, 60))

plt_depth = plot_depth_img(img, title="Simuliertes Tiefenbild", colorbar_title="Distanz [m]", xlabel="x-Pixel", ylabel="y-Pixel")
plot_depth_img(logdensityof.(my_pixel_dist.(μ, o), img) .|> exp; colorbar_title="probability density", reverse=false, value_to_typemax=1)
# Shorter differences should result in higher logdensity. Keep in mind the mixture.

dist_is(μ) = ValidPixel(μ, KernelNormal(μ, 0.1f0))
dist_not(μ) = ValidPixel(μ, truncated(KernelExponential(1.0f0), nothing, μ))
ia = ImageAssociation(dist_is, dist_not, fill(0.5f0, 100, 100), μ)
ℓ = logdensityof(ia, img)
pyplot()
plt_o = plot_prob_img(ℓ; title="Objektzugehörigkeit", colorbar_title="Wahrscheinlichkeit [0,1]", xlabel="x-Pixel", ylabel="y-Pixel")

plot(plt_depth, plt_o; layout=(1, 2))
