# A logistic map but not with DifferentialEquations.jl as a backend
using MCBB
using DifferentialEquations
using Distributions
using LightGraphs
using Clustering


function logistic_custom(u0, p::logistic_parameters, tspan)
    tmin=tspan[1]+1
    tmax=tspan[2]
    tsteps = tspan[2] - tspan[1]

    x = zeros(tsteps+1)

    x[1] = u0[1]
    for it=tmin:tmax
        x[it] = p.r*x[it-1]*(1 - x[it-1])
    end
    x
end

rdist = Uniform(2.5,4)
r = ()->rand(rdist)
pars = logistic_parameters(r())
icdist = Uniform(0.1,0.99)
ic_ranges = ()->rand(icdist)
N_ic = 500
dp = CustomProblem(logistic_custom, [ic_ranges()], (1,1000), pars)
tail_frac = 0.8
log_emcp = CustomMCBBProblem(dp, ic_ranges, N_ic, pars, (:r, r), eval_ode_run, tail_frac)
log_sol = solve(log_emcp)

# analysis
#D = distance_matrix(log_sol);

D = @time distance_matrix(log_sol, log_emcp, [1,0.75,0.5,1]);

fdist = k_dist(D,4);

db_eps = 0.065
db_res = dbscan(D,db_eps,4)

cluster_meas = cluster_means(log_sol,db_res);
cluster_n = cluster_n_noise(db_res);
cluster_members = cluster_membership(log_emcp,db_res,0.005,0.001);


true
