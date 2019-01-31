using HighBifLib

using LightGraphs
using Clustering
using DifferentialEquations
using Distributions
using StatsBase

N = 15
K = 0.5
nd = Normal(0.5, 0.2)
w_i_par = rand(nd,N)

net = erdos_renyi(N, 0.25)
A = adjacency_matrix(net)

ic = zeros(N)
ic_dist = Uniform(-pi,pi)
kdist = Uniform(0,5)
ic_ranges = ()->rand(ic_dist)
N_ics = 100
K_range = (i)->rand(kdist)
std_range = (i)->rand(Uniform(0.0,0.75))

new_kura_par(old_par; K=1, std=0.2) = kuramoto_network_parameters(K, rand(Normal(0.5, std), N), N, A)
par_var = MultiDimParameterVar([OneDimParameterVar(:K,K_range),OneDimParameterVar(:std,std_range)], new_kura_par)

pars = kuramoto_network_parameters(K, w_i_par, N, A)

# base problem
rp = ODEProblem(kuramoto_network, ic, (0.,3000.), pars)

# we also calculate the order parameter, we won't use it for clustering, but we'll use it as a check
function k_order_parameter(u::AbstractArray)
    uend = u[:,end]
    N = length(uend)
    1. /N*sqrt((sum(sin.(uend)))^2+(sum(cos.(uend)))^2)
end

function eval_ode_run_kura(sol, i)
    (N_dim, __) = size(sol)
    state_filter = collect(1:N_dim)
    eval_funcs = [mean, std]
    global_eval_funcs = [k_order_parameter]
    eval_ode_run(sol, i, state_filter, eval_funcs, global_eval_funcs, cyclic_setback=true)
end

tail_frac = 0.9 #
ko_mcp = BifAnaMCProblem(rp, ic_ranges, N_ics, pars, par_var, eval_ode_run_kura, tail_frac)
kosol = solve(ko_mcp)

D_k = distance_matrix(kosol, ko_mcp, [1,0.75,0.,1.,1.]); # no weight on the order_parameter and kl div

db_eps = 110 # we found that value by scanning manually
db_res = dbscan(D_k,db_eps,4)
cluster_members = cluster_membership(ko_mcp,db_res,[0.5,0.05],[0.2,0.05]);

true
