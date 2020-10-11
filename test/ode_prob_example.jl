# simple Kuramoto example
using MCBB
using DifferentialEquations
using Distributions
using StatsBase
using Clustering

# common setup
N = 5
K = 0.5
nd = Normal(0.5, 0.01) # distribution for eigenfrequencies # mean = 0.5Hz, std = 0.5Hz
w_i_par = rand(nd,N)
A = ones(N,N)
ic = zeros(N)
ic_dist = Uniform(-pi,pi)
kdist = Uniform(0,5)
pars = kuramoto_network_parameters(K, w_i_par, N, A)
rp = ODEProblem(kuramoto_network, ic, (0.,40.), pars)

# range + range
ic_ranges = [0.:0.5:1.5 for i=1:N]
k_range = 1.:0.5:2.
tail_frac = 0.9 #

function my_eval_ode_run(sol, i)
    N_dim = length(sol.prob.u0)
    state_filter = collect(1:N_dim)
    eval_funcs = [mean, std]
    eval_ode_run(sol, i, state_filter, eval_funcs)
end

ko_emcp = DEMCBBProblem(rp, ic_ranges, pars, (:K, k_range), eval_ode_run, tail_frac)
ko_sol = solve(ko_emcp, Rosenbrock23())

# random + range
ic_ranges = ()->rand(ic_dist)
k_range = 1.:0.5:3.
N_ics = 150

ko_emcp = DEMCBBProblem(rp, ic_ranges, N_ics, pars, (:K, k_range), my_eval_ode_run, tail_frac)
ko_sol = solve(ko_emcp)

# define a random array
ic_array = ()->rand(ic_dist, N)
k_range = ()->rand(kdist)
ko_emcp = DEMCBBProblem(rp, ic_ranges, N_ics, pars, (:K, k_range), eval_ode_run, tail_frac)
ko_sol = solve(ko_emcp)

function eval_ode_run_all_test(sol, i)
    N_dim = length(sol.prob.u0)
    state_filter = collect(1:N_dim)
    eval_funcs = [mean, std]
    matrix_eval_funcs = [correlation_ecdf]
    global_eval_funcs = [(x)->sum(x)]
    eval_ode_run(sol, i, state_filter, eval_funcs, matrix_eval_funcs, global_eval_funcs)
end


# random + random
ic_ranges = [()->rand(ic_dist)]
k_range = (i)->rand(kdist)

ko_emcp = DEMCBBProblem(rp, ic_ranges, N_ics, pars, (:K, k_range), eval_ode_run_all_test, tail_frac)
ko_sol = solve(ko_emcp)

D = distance_matrix(ko_sol, ko_emcp, [1.,0.5,1.,1.,1], histograms=true, matrix_distance_func=wasserstein_histogram_distance,nbin_default=5);

D = distance_matrix(ko_sol, ko_emcp, [1.,0.5,1.,1.,1], matrix_distance_func=wasserstein_histogram_distance,nbin_default=5);
k = 4
#fdist = k_dist(D,k);



# analysis
db_eps = 1000
db_res = dbscan(D,db_eps,k)
cluster_meas = cluster_means(ko_sol,db_res);
cluster_n = cluster_n_noise(db_res);
cluster_members = cluster_membership(ko_emcp,db_res,0.2,0.05);
res = cluster_measures(ko_emcp, ko_sol, db_res, 0.2, 0.05);
res_hist =cluster_measures_sliding_histograms(ko_emcp, ko_sol, db_res, 1, 0.2, 0.05);
cisc = ClusterICSpaces(ko_emcp, ko_sol, db_res)

function eval_ode_run_filter(sol, i)
    N_dim = length(sol.prob.u0)
    state_filter = collect(2:4)
    eval_funcs = [mean, std]
    matrix_eval_funcs = [correlation_ecdf]
    global_eval_funcs = [(x)->sum(x)]
    eval_ode_run(sol, i, state_filter, eval_funcs, matrix_eval_funcs, global_eval_funcs)
end

ko_emcp = DEMCBBProblem(rp, ic_ranges, N_ics, pars, (:K, k_range), eval_ode_run_filter, tail_frac)
ko_sol = solve(ko_emcp)

D = distance_matrix(ko_sol, ko_emcp, [1.,0.5,1.,1.,1], histograms=true, matrix_distance_func=wasserstein_histogram_distance);

D_sparse = MCBB.distance_matrix_sparse(ko_sol, ko_emcp, [1.,0.5,1.,1.,1], histograms=true, matrix_distance_func=wasserstein_histogram_distance, sparse_threshold=1.)

db_eps = 100
db_res = dbscan(D,db_eps,k)
cluster_meas = cluster_means(ko_sol,db_res);
cluster_n = cluster_n_noise(db_res);
cluster_members = cluster_membership(ko_emcp,db_res,0.2,0.05);
res = cluster_measures(ko_emcp, ko_sol, db_res, 0.2, 0.05);
res_hist =cluster_measures_sliding_histograms(ko_emcp, ko_sol, db_res, 1, 0.2, 0.05, state_filter=1:2);
cisc = ClusterICSpaces(ko_emcp, ko_sol, db_res)






true
