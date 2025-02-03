# About
Evaluates sampling-based Bayesian inference (different variants of MCMC, SMC) for the 6D pose estimation of objects using depth images and CAD models only.
This code has been produced during while writing my Ph.D. (Dr.-Ing.) thesis at the Institut of Automatic Control, RWTH Aachen University.
If you find it helpful for your research please cite this:
> T. Redick, „Bayesian inference for CAD-based pose estimation on depth images for robotic manipulation“, RWTH Aachen University, 2024. doi: [10.18154/RWTH-2024-04533](https://doi.org/10.18154/RWTH-2024-04533).

I submitted my results of the best performing SMC sampler to the [BOP benchmark](https://bop.felk.cvut.cz/home/) with two different time budgets per pose estimate:
* 1 second: https://bop.felk.cvut.cz/method_info/458/
* 0.5 seconds: https://bop.felk.cvut.cz/method_info/457/

# Required Julia packages
These packages will be installed using the `[sources]` section in `Project.toml`.
* https://github.com/rwth-irt/BayesNet.jl - Type stable implementation of a Bayesian network.
* https://github.com/rwth-irt/KernelDistributions.jl - Subset of Distributions.jl which can be used in CUDA kernels.
* https://github.com/rwth-irt/PoseErrors.jl - 6D pose error metrics from [BOP Challenge](https://bop.felk.cvut.cz/home/)
* https://github.com/rwth-irt/SciGL.jl - Efficient rendering in OpenGL and CUDA interop for julia
* https://github.com/rwth-irt/BlenderProc.DissTimRedick - BlenderProc setup to generate the synthetic datasets from my dissertation. 

# Obtaining the Datasets
With the exception of the STERI dataset, all datasets can be obtained from the [BOP benchmark](https://bop.felk.cvut.cz/home/).
The STERI dataset contains proprietary CAD models, which I cannot make available.

My own data sets can be obtained from me [RWTH publications soon](TODO ongoing request).
- **BOP Datasets** includes the synthetically generated training datasets generated with [BlenderProc](https://github.com/DLR-RM/BlenderProc). These can be used for parameter tuning.
- **rosbags** contains the rosbags to evaluate the particle filter. Ground truth data has been obtained using the forward kinematics of the robot holding the camera.

Extract the datasets into the `data/bop` directory, e.g. `data/bop/itodd`.

# Notes on the Project Structure
I use [DrWatson.jl](https://juliadynamics.github.io/DrWatson.jl/stable/) to manage the experiments and follow their [project setup conventions](https://juliadynamics.github.io/DrWatson.jl/stable/project/).
For example, the `data`, `scripts`, and `src` folders are the result of this convention.

# Note on recent CUDA.jl versions
v5.0.0 introduced some changes which negatively impact performance
* when assigning `prior_o[mask_img] .= ...` an "attempt to release free memory error occurs"
* benchmark `simple_posterior` vs. `smooth_posterior` after upgrading
