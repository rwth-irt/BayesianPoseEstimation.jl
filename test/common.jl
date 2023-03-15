# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using MCMCDepth
using Test

@testset "to_pose" begin
    # Single pose
    t = rand(Float32, 3)
    r = rand(Quaternion{Float32})
    pose = @inferred to_pose(t, r)

    # Multiple poses
    t5 = rand(Float32, 3, 5)
    r5 = rand(Quaternion{Float32}, 5)
    pose55 = @inferred to_pose(t5, r5)
    pose51 = @inferred to_pose(t5, r)
    pose15 = @inferred to_pose(t, r5)

    @test reduce(&, [p.rotation != pose.rotation for p in pose55])
    @test reduce(&, [p.translation != pose.translation for p in pose55])

    @test reduce(&, [p.rotation == pose.rotation for p in pose51])
    @test reduce(&, [p.translation != pose.translation for p in pose51])

    @test reduce(&, [p.translation == pose.translation for p in pose15])
    @test reduce(&, [p.rotation != pose.rotation for p in pose15])
end
