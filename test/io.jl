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

function eval_ode_run_all_test(sol, i)
    N_dim = length(sol.prob.u0)
    state_filter = collect(1:N_dim)
    eval_funcs = [mean, std]
    matrix_eval_funcs = [correlation_ecdf]
    global_eval_funcs = [(x)->sum(x)]
    eval_ode_run(sol, i, state_filter, eval_funcs, matrix_eval_funcs, global_eval_funcs)
end

ko_emcp = DEMCBBProblem(rp, ic_array, N_ics, pars, (:K, k_range), eval_ode_run_all_test, tail_frac)
ko_sol = solve(ko_emcp)

ko_sol_backup = ko_sol

MCBB.save(ko_emcp, "prob-test-save.jld2")
MCBB.save(ko_sol, "sol-test-save.jld2")


ko_emcp = nothing
ko_sol = nothing

ko_emcp = MCBB.load_prob("prob-test-save.jld2", rp, eval_ode_run_all_test, (:K, k_range))
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

D = distance_matrix(ko_sol, ko_emcp, [1.,0.5,1.,1.,1], histograms=true, matrix_distance_func=wasserstein_histogram_distance);
#println(D[1:50,1])
MCBB.save(D, "D-test-save.jld2")

D = nothing

D_2 = distance_matrix(ko_sol, ko_emcp, [1.,0.5,1.,1.,1], histograms=true, matrix_distance_func=wasserstein_histogram_distance);
#println(D_2[1:50,1])


D = MCBB.load_D_hist("D-test-save.jld2", matrix_distance_func=wasserstein_histogram_distance)
#println(D[1:50,1])#3println(D_2.==D)
if sum(D_2 .== D) == ((N_ics*N_ics)+sum(isnan.(D)))
    test_result = true
else
    test_results = false
    println("distance unequal")
end

D = distance_matrix(ko_sol, ko_emcp, [1.,0.5,1.,1.,1], matrix_distance_func=wasserstein_histogram_distance);

MCBB.save(D, "D-test-save.jld2")

D = nothing

D_2 = distance_matrix(ko_sol, ko_emcp, [1.,0.5,1.,1.,1], matrix_distance_func=wasserstein_histogram_distance);

D = MCBB.load_D("D-test-save.jld2", matrix_distance_func=wasserstein_histogram_distance)

if sum(D_2 .== D) == ((N_ics*N_ics)+sum(isnan.(D)))
    test_result = true
else
    test_results = false
    println("distance unequal")
end


rm("prob-test-save.jld2")
rm("sol-test-save.jld2")
rm("D-test-save.jld2")

true
