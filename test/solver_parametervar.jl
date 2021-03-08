using MCBB
using DifferentialEquations
using Distributions
using StatsBase
using Clustering

# common setup
N = 20
K = 0.5
nd = Normal(0.5, 0.005) # distribution for eigenfrequencies # mean = 0.5Hz, std = 0.5Hz
w_i_par = rand(nd,N)
A = ones(N,N)
ic = zeros(N)
ic_dist = Uniform(-pi,pi)
kdist = Uniform(0,5)
pars = kuramoto_network_parameters(K, w_i_par, N, A)
rp = ODEProblem(kuramoto_network, ic, (0.,40.), pars)


tail_frac = 0.9 #
ics = ()->rand(ic_dist)

parvar = SolverParameterVar(:reltol, ()->rand(Uniform(1e-6,1e-3)))

N_ics = 500

ko_emcp = DEMCBBProblem(rp, ics, N_ics, pars, parvar, eval_ode_run, tail_frac)

ko_sol = solve(ko_emcp, Tsit5())

return true
