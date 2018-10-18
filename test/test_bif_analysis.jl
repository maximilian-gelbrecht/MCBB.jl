# test the BifAnalysisProblem

using HighBifLib
using Distributions
using LightGraphs

pars = logistic_parameters(2.1)
dp = DiscreteProblem(logistic, [0.5], (0.,100.), pars)

r_range = 2:0.1:4
bap1 = BifAnalysisProblem(dp, (:r,r_range), eval_ode_run)

r_fun = (r_old)->r_old+0.05*rand()
bap2 = BifAnalysisProblem(dp, (:r,r_fun), 21, eval_ode_run, [0.001,0.999])

s1 = solve(bap1)
s2 = solve(bap2)

N = 6
K = 0.01
nd = Normal(0.5, 0.05) # distribution for eigenfrequencies # mean = 0.5Hz, std = 0.5Hz
w_i_par = rand(nd,N)
net = erdos_renyi(N, 0.2)
A = adjacency_matrix(net)
ic_dist = Uniform(-pi,pi)
ic = rand(ic_dist,N)

pars = kuramoto_network_parameters(K, w_i_par, N, A)
rp = ODEProblem(kuramoto_network, ic, (0.,100.), pars)

K_range = 0.01:0.1:2
kap2 = BifAnalysisProblem(rp, (:K,K_range), eval_ode_run)

K_fun = (k_old)->k_old+0.05*rand()
kap2 = BifAnalysisProblem(rp, (:K,K_fun), 21, eval_ode_run)

true
