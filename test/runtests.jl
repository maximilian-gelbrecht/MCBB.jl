#!/usr/bin/env julia

#Start Test Script
using MCBB
using Test

# Run tests
println("Test ODE Problems")
@time @test include("ode_prob_example.jl")
println("Test BifAnalysisProblem")
@time @test include("test_bif_analysis.jl")
println("Test Discrete Problems")
@time @test include("discrete_prob_example.jl")

println("Test Multidim Setups")
@time @test include("multi_dim_example.jl")
println("Test Custom Problems")
@time @test include("custom_problem_example.jl")
