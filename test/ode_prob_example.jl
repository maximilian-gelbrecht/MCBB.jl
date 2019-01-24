# simple Kuramoto example
using HighBifLib
using DifferentialEquations
using Distributions
using LightGraphs
using Clustering

# common setup
N = 6
K = 0.5
nd = Normal(0.5, 0.05) # distribution for eigenfrequencies # mean = 0.5Hz, std = 0.5Hz
w_i_par = rand(nd,N)
net = erdos_renyi(N, 0.2)
A = adjacency_matrix(net)
ic = zeros(N)
ic_dist = Uniform(-pi,pi)
kdist = Uniform(0,10)
pars = kuramoto_network_parameters(K, w_i_par, N, A)
rp = ODEProblem(kuramoto_network, ic, (0.,100.), pars)

# range + range
ic_ranges = [0.:0.5:1.5 for i=1:N]
k_range = 1.:0.5:3.
tail_frac = 0.9 #

ko_emcp = BifAnaMCProblem(rp, ic_ranges, pars, (:K, k_range), eval_ode_run, tail_frac)
ko_sol = solve(ko_emcp)

# random + range
ic_ranges = ()->rand(ic_dist)
k_range = 1.:0.5:3.
N_ics = 20

ko_emcp = BifAnaMCProblem(rp, ic_ranges, N_ics, pars, (:K, k_range), eval_ode_run, tail_frac)
ko_sol = solve(ko_emcp)


# random + random
ic_ranges = [()->rand(ic_dist)]
k_range = (i)->rand(kdist)

ko_emcp = BifAnaMCProblem(rp, ic_ranges, N_ics, pars, (:K, k_range), eval_ode_run, tail_frac)
ko_sol = solve(ko_emcp)

D = distance_matrix(ko_sol);

D = distance_matrix(ko_sol, parameter(ko_emcp));
k = 4
fdist = k_dist(D,k);

# analysis
db_eps = 1
db_res = dbscan(D,db_eps,k)
cluster_meas = cluster_means(ko_sol,db_res);
cluster_n = cluster_n_noise(db_res);
cluster_members = cluster_membership(ko_emcp,db_res,0.2,0.05);
(p_win, cluster_measures_dim, cluster_measures_global) = cluster_measures(ko_emcp, ko_sol, db_res, 0.2, 0.05);
cisc = ClusterICSpaces(ko_emcp, ko_sol, db_res)

true
