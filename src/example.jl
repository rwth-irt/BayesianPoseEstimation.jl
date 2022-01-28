# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using MCMCDepthTuring
using StatsPlots
using Turing

d = CircularUniform()
x = rand(d)
pdf(d, x)

b = bijector(d)
y = b(-1)
b⁻¹ = Inverse(b)
b⁻¹(y)
l = logpdf_with_trans(d, x, true)

@model function m1()
  x = Matrix{Float32}(undef, 3, 3)
  y = Matrix{Float32}(undef, 3, 3)
  for i in eachindex(x)
    x[i] ~ CircularUniform()
    y[i] ~ Normal(x[i], 1)
  end
end

# Handling the chain for evaluation
chain = sample(m1(), Prior(), 10000)
names = namesingroup(chain, :x)
a = chain[names] |> Array
length(a[:, 1])
b = [reshape(a[i, :], (3, 3)) for i in 1:length(a[:, 1])]
mean(b)
c = reshape(transpose(a), 3, 3, :)
dropdims(mean(c, dims = 3), dims = 3)
mm = [rand(MixtureModel([Normal(0, 1), MixtureModel([Normal(10, 1), Normal(20, 1)], [0.9, 0.1])], [0.9, 0.1])) for i in 1:1000]

MyMixture(μ) = MixtureModel([Normal(μ, 1), Exponential(1)], [0.9, 0.1])

# What about the mixtures?
@model function mix_model(y)
  μ ~ filldist(Exponential(1), 10, 10)
  if y === missing
    y = Matrix{Float64}(undef, 10, 10)
  end
  I = Identity(10)
  y = vec(y)
  μ = vec(μ)
  y .~ MyMixture.(μ)
end

y = rand(MixtureModel([Normal(10, 1), Exponential(1)], [0.9, 0.1]), 100)
mix_model(missing)()
sample(mix_model(y), MH(), MCMCThreads(), 10, 4)
density(y)
