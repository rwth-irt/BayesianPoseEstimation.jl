# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using MCMCDepth
using Test

@testset "Parameters" begin
    p = Parameters()
    p = @set p.float_type = Float16
    @test p.float_type == Float16

    @test p.min_depth isa Float16
    @test p.max_depth isa Float16

    @test p.pixel_σ isa Float16
    @test p.pixel_θ isa Float16
    @test p.mix_exponential isa Float16

    @test p.mean_t |> eltype == Float16
    @test p.σ_t |> eltype == Float16
    @test p.proposal_σ_t |> eltype == Float16
    @test p.proposal_σ_r |> eltype == Float16
end
