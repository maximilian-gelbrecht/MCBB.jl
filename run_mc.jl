



   
cluster = true
       
if isdefined(:cluster)
    using ClusterManagers
    N_tasks = parse(Int, ARGS[1])
    N_worker = N_tasks - 1
    ddprocs(SlurmManager(N_worker)) # -1 cause one process is the master
    @everywhere include("/p/tmp/maxgelbr/julia-test/montecarlo/myLib.jl")
else
    addprocs(3)
    @everywhere include("myLib.jl")
end

using JLD2, FileIO
@everywhere using DifferentialEquations
@everywhere using Distributions
@everywhere using myLib

   



   
       
N = 500
K = 0.5
nd = Normal(0.5, 0.5)   # mean = 0.5Hz, std = 0.5Hz
ud = Uniform(-pi,pi)
pars = kuramoto_parameters(K, rand(nd, N), N)
rp = ODEProblem(kuramoto, rand(ud, N), (0.,100.), pars)

K = collect(0.1:0.025:10)
N_mc = size(K)[1]

# For MonteCarlo simulations we have to build a new problem every time. This takes a base problem,
# and gives back a new one, to illustrate, the following just gives back the same problem:
@everywhere inc_coupling_problem = (prob,i,repeat) -> ODEProblem(prob.f, rand(ud, N), prob.tspan, kuramoto_parameters(K[i],rand(nd,N),N))
# The parameter prob is the base problem we are varying.
# The parameter i is the index of the Monte Carlo run we are generating the problem for.
# The repeat parameter indicates whether we are rerunning an experiment. Ignore it for now.
@everywhere output_func = (sol, i) -> (order_parameter(sol[:,end], N), false)
# define the MC Problem
mcp = MonteCarloProblem(rp, prob_func=inc_coupling_problem, output_func=output_func)

# solve it, the :pmap symbol defines the way the problem is parallized by julia
#sol_mc = @time solve(mcp, num_monte=2, parallel_type=:pmap)
sol_mc = @time solve(mcp, num_monte=N_mc, collect_result = Val{true}, parallel_type=:parfor)
#sol_mc = @time solve(mcp, num_monte=2, parallel_type=:none)
# save the solutions, analyze them later
#@save "mc_roes_coupled.jld2" sol_mc


   



   
       
writedlm("orderparameter.txt", sol_mc[:])


   

