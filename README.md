# CUDA version
v5.0.0 introduced some changes which negatively impact performance
* when assigning `prior_o[mask_img] .= ...` an "attempt to release free memory error occurs"
* benchmark `simple_posterior` vs. `smooth_posterior` after upgrading
