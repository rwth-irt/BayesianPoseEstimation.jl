# About
Evaluates sampling-based Bayesian inference (different variants of MCMC, SMC) for the 6D pose estimation of objects using depth images and CAD models only.
This code has been produced during while writing my Ph.D. (Dr.-Ing.) thesis at the institut of automatic control, RWTH Aachen University.
If you find it helpful for your research please cite this:
> T. Redick, „Bayesian inference for CAD-based pose estimation on depth images for robotic manipulation“, RWTH Aachen University, 2024. doi: [10.18154/RWTH-2024-04533](https://doi.org/10.18154/RWTH-2024-04533).

I submitted my results of the best performing SMC sampler to the [BOP benchmark](https://bop.felk.cvut.cz/home/) with two different time budgets per pose estimate:
* 1 second: https://bop.felk.cvut.cz/method_info/458/
* 0.5 seconds: https://bop.felk.cvut.cz/method_info/457/

# Required Julia packages
Since this code has been written, before `[sources]` has been [supported in Project.toml](https://github.com/JuliaLang/Pkg.jl/pull/3783#issuecomment-2138812311) and I didn't register my standalone Julia packages, you might need these in manually:
* https://github.com/rwth-irt/BayesNet.jl - Type stable implementation of a Bayesian network.
* https://github.com/rwth-irt/KernelDistributions.jl - Subset of Distributions.jl which can be used in CUDA kernels.
* https://github.com/rwth-irt/PoseErrors.jl - 6D pose error metrics from [BOP Challenge](https://bop.felk.cvut.cz/home/)
* https://github.com/rwth-irt/SciGL.jlhttps://github.com/rwth-irt/SciGL.jl - Efficient rendering in OpenGL and CUDA interop for julia
* https://github.com/rwth-irt/BlenderProc.DissTimRedick - BlenderProc setup to generate the synthetic datasets from my dissertation. 

# Note on recent CUDA.jl versions
v5.0.0 introduced some changes which negatively impact performance
* when assigning `prior_o[mask_img] .= ...` an "attempt to release free memory error occurs"
* benchmark `simple_posterior` vs. `smooth_posterior` after upgrading
