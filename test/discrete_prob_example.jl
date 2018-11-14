using HighBifLib
using DifferentialEquations
using Distributions
using Clustering
# ranges
r = 3:0.05:3.3
pars = logistic_parameters(r[1])
ic_ranges = [0.1:0.3:0.9]
dp = DiscreteProblem(logistic, ic_ranges[1][1], (0.,5000.), pars)
tail_frac = 0.8
log_emcp = BifAnaMCProblem(dp, ic_ranges, pars, (:r, r), eval_ode_run, tail_frac)
log_sol = solve(log_emcp)

# random+range
r =  3:0.05:3.3
pars = logistic_parameters(r[1])
icdist = Uniform(0.1,0.9)
ic_ranges = ()->rand(icdist)
N_ic = 20
dp = DiscreteProblem(logistic, ic_ranges(), (0.,5000.), pars)
log_emcp = BifAnaMCProblem(dp, ic_ranges, N_ic, pars, (:r, r), eval_ode_run, tail_frac)
log_sol = solve(log_emcp)

# random+random
rdist = Uniform(3,3.3)
r = ()->rand(rdist)
pars = logistic_parameters(r())
icdist = Uniform(0.1,0.9)
ic_ranges = ()->rand(icdist)
N_ic = 20
dp = DiscreteProblem(logistic, ic_ranges(), (0.,5000.), pars)
tail_frac = 0.8
log_emcp = BifAnaMCProblem(dp, ic_ranges, N_ic, pars, (:r, r), eval_ode_run, tail_frac)
log_sol = solve(log_emcp)

# analysis
D = distance_matrix(log_sol);

D = distance_matrix(log_sol, parameter(log_emcp));

fdist = k_dist(D,4);

db_eps = 150
db_res = dbscan(D,db_eps,4)

cluster_meas = cluster_means(log_sol,db_res);
cluster_n = cluster_n_noise(db_res);
cluster_members = cluster_membership(parameter(log_emcp),db_res,0.2,0.05);

true
