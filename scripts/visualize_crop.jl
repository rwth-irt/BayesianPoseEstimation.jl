# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using FileIO
using MCMCDepth
using PoseErrors
import CairoMakie as MK

diss_defaults()

parameters = Parameters()
resolutions = [15, 25, 50]

fig = MK.Figure(resolution=(DISS_WIDTH, 0.3 * DISS_WIDTH))
for (idx, width) in enumerate(resolutions)
    @reset parameters.width = width
    @reset parameters.height = parameters.width
    df = gt_targets(joinpath("data", "bop", "lm", "test"), 2)
    row = df[101, :]
    color_img = load_color_image(row, parameters.img_size...)
    depth_img = load_depth_image(row, parameters.img_size...)

    ax = img_axis(fig[1, idx]; aspect=1)
    # Plot the image as background
    MK.image!(ax, color_img; aspect=1, interpolate=false, rasterize=true)
end
fig
MK.save(joinpath("plots", "crop_resolutions.pdf"), fig)