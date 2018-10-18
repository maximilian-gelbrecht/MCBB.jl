#!/usr/bin/env julia

#Start Test Script
using HighBifLib
using Base.Test

# Run tests

tic()
println("Test Discrete Problems")
@time @test include("discrete_prob_example.jl")
println("Test ODE Problems")
@time @test include("ode_prob_example.jl")
println("Test BifAnalysisProblem")
@time @test include("test_bif_analysis.jl")
toc()
