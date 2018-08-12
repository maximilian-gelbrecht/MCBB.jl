



   
cluster = true
       
if isdefined(:cluster)
    using ClusterManagers
    N_tasks = parse(Int, ARGS[1])
    N_worker = N_tasks 
    addprocs(SlurmManager(N_worker))
    @everywhere include("/p/tmp/maxgelbr/code/HighBifLib.jl/HighBifLib.jl")
else

    addprocs(1)
    @everywhere include("HighBifLib.jl")
end

@everywhere using LSODA
@everywhere using LightGraphs
using JLD2, FileIO
@everywhere using DifferentialEquations
@everywhere using Distributions
@everywhere using HighBifLib  

# these imports invoke a lot of warnings when executed with multiple processes
# this seems to me to be more a bug of julia than an actual problem with this code
# imported on a single process there are no warnings

   









   
       
@everywhere r = 2.4:0.005:4
@everywhere pars = logistic_parameters(r[1])
@everywhere ic_ranges = [0.1:0.01:0.9]
@everywhere dp = DiscreteProblem(logistic, ic_ranges[1][1], (0.,500.), pars)
@everywhere (ic_r_prob, ic_par, N_mc) = setup_ic_par_mc_problem(dp, ic_ranges, pars, (:r, r))

log_mcp = MonteCarloProblem(dp, prob_func=ic_r_prob, output_func=eval_ode_run)
tail_frac = 0.8
log_emcp = EqMCProblem(log_mcp, N_mc, tail_frac)
log_sol = solve(log_emcp)

if defined(:cluster)
    @save "mc_log.jld2" log_sol
end 


   



























