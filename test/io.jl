# simple Kuramoto example
using MCBB
using DifferentialEquations
using Distributions
using StatsBase
using JLD2

# common setup
N = 5
K = 0.5
nd = Normal(0.5, 0.05) # distribution for eigenfrequencies # mean = 0.5Hz, std = 0.5Hz
w_i_par = rand(nd,N)
A = ones(N,N)
ic = zeros(N)
ic_dist = Uniform(-pi,pi)
kdist = Uniform(0,5)
pars = kuramoto_network_parameters(K, w_i_par, N, A)
rp = ODEProblem(kuramoto_network, ic, (0.,40.), pars)

N_ics = 500
# define a random array
ic_array = ()->rand(ic_dist, N)
k_range = ()->rand(kdist)
tail_frac = 0.8
ko_emcp = DEMCBBProblem(rp, ic_array, N_ics, pars, (:K, k_range), eval_ode_run, tail_frac)
ko_sol = solve(ko_emcp)

ko_sol_backup = ko_sol

MCBB.save(ko_emcp, "prob-test-save.jld2")
MCBB.save(ko_sol, "sol-test-save.jld2")

ko_emcp = nothing
ko_sol = nothing

ko_emcp = MCBB.load_prob("prob-test-save.jld2", rp, eval_ode_run, (:K, k_range))
ko_sol_new = solve(ko_emcp)
ko_sol = MCBB.load_sol("sol-test-save.jld2")

test_result = true
for i=1:N_ics
    if ko_sol.sol[i] != ko_sol_backup.sol[i]
        println("unequal")
        test_result = false
    end
    if ko_sol.sol[i] != ko_sol_new.sol[i]
        println("unequal")
        test_result = false
    end
end


rm("prob-test-save.jld2")
rm("sol-test-save.jld2")

true
