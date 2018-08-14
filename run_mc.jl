




cluster = true

if isdefined(:cluster)
    using ClusterManagers
    N_tasks = parse(Int, ARGS[1])
    N_worker = N_tasks
    addprocs(SlurmManager(N_worker))
    @everywhere include("/p/tmp/maxgelbr/code/HighBifLib.jl/HighBifLib.jl")
else
    #addprocs(1)
    @everywhere include("HighBifLib.jl")
    import Plots
end

@everywhere using LSODA
@everywhere using LightGraphs
using JLD2, FileIO, Clustering
@everywhere using DifferentialEquations
@everywhere using Distributions
@everywhere using HighBifLib

# these imports invoke a lot of warnings when executed with multiple processes
# this seems to me to be more a bug of julia than an actual problem with this code
# imported on a single process there are no warnings





























@everywhere N = 5

@everywhere k = 2
@everywhere p = 0.2
@everywhere net = watts_strogatz(N, k, p)

#@everywhere A = [0 1 1
#    1 0 1
#    1 1 0 ]
#@everywhere net = Graph(A)

@everywhere L = laplacian_matrix(net)

@everywhere a = ones(N).*0.2
@everywhere b = ones(N).*0.2
@everywhere c = ones(N).*7.0

tail_frac = 0.8

# for reference get the synchronizable range of Ks
evals = eig(full(L))[1]
evals = sort!(evals[evals .> 1e-5])
lambda_min = evals[end]
lambda_max = evals[1]
K_sync = (0.1232/lambda_min, 4.663/lambda_max)







#@everywhere K_range = [0.002, 0.004, 0.006, 0.008, 0.010, 0.03, 0.05, 0.1, 0.2, 0.5, 1, 2, 4]
@everywhere K_range = [0.002, 0.004, 0.03, 0.5, 1, 4]
#@everywhere K_range = [0.002, 0.005, 0.01, 0.025, 0.05, 0.07, 0.5, 1, 2]
#@everywhere K_range = [0.005, 0.007, 0.009, 0.011, 0.013, 0.016, 0.02, 0.03, 0.04, 0.9, 1.5, 2]
@everywhere ic_gen_xy = Uniform(-15.,15.)
@everywhere ic_gen_z = Uniform(-5.,20.)

@everywhere ic_gens = [()->rand(ic_gen_xy), ()-> rand(ic_gen_xy), ()->rand(ic_gen_z)]
@everywhere N_ic = 5


@everywhere rp = ODEProblem(roessler_network, zeros(3*N), (0.,100.), roessler_parameters(a,b,c,0.05,L,N))
@everywhere (ic_coupling_problem, ic_par, N_mc) = setup_ic_par_mc_problem(rp, ic_gens, N_ic, roessler_parameters(a,b,c,0.05,L,N),(:K,K_range))








rn_mcp = MonteCarloProblem(rp, prob_func=ic_coupling_problem, output_func=eval_ode_run_inf)
rn_emcp = EqMCProblem(rn_mcp, N_mc, tail_frac)
@time rn_sol = solve(rn_emcp)

if isdefined(:cluster)
    @save "mc_roes_sol_inf.jld2" rn_sol
    @save "mc_roes_ic_par_inf.jld2" ic_par
end
