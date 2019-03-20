# simple Kuramoto example
using MCBB
using DifferentialEquations
using Distributions
using LightGraphs
using StatsBase
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

function my_eval_ode_run(sol, i)
    N_dim = length(sol.prob.u0)
    state_filter = collect(1:N_dim)
    eval_funcs = [mean, std]
    eval_ode_run(sol, i, state_filter, eval_funcs)
end

function meval_ode_run(sol, i, state_filter::Array{Int64,1}, eval_funcs::AbstractArray, global_eval_funcs::AbstractArray; failure_handling::Symbol=:None, cyclic_setback::Bool=false, replace_inf=nothing, flag_past_measures=false)
    N_dim = length(sol.prob.u0)
    N_dim_measures = length(eval_funcs)  # mean and var are always computed

    if N_dim_measures < 1
        error("No per dimension measures")
    end

    N_dim_global_measures = length(global_eval_funcs)

    if failure_handling==:None
        failure_handling=:None # do nothing
    elseif failure_handling==:Inf
        if (sol.retcode == :DtLessThanMin) | (sol.retcode == :Unstable)
            # in case it is so unstable that the solution is empty as no results are returned
            if length(sol) == 0
                inf_flag = true
            else
                last = sol.u[end]
                inf_flag = false
                for i=1:N_dim
                    if abs(last[i]) > 1e11
                        inf_flag = true
                    end
                end
            end
            if inf_flag
                dim_measures = [Inf.*ones(N_dim) for i=1:N_dim_measures]
                global_measures = [Inf for i=1:N_dim_global_measures]
                return (tuple(dim_measures...,global_measures...),false)
            else
                @warn "Failure Handling Warning, DtLessThanMin but Solution not diverging."
            end
        end
    elseif failure_handling==:Repeat
        if (sol.retcode != :Success) & (sol.retcode != :Default)
            return ((),true)
        end
    else
        error("failure_handling symbol not known")
    end
    (N_dim, N_t) = size(sol)

    if replace_inf != nothing
        inf_ind = isinf.(sol)
        pinf_ind = inf_ind .& (sol .> 0)
        minf_ind = inf_ind .& (.~pinf_ind)

        for i_dim=1:N_dim
            for it=1:N_t
                if pinf_ind[i_dim,it]
                    sol.u[it,:][i_dim] = replace_inf
                end
                if minf_ind[i_dim,it]
                    sol.u[it,:][1][i_dim] = -1 * replace_inf
                end
            end
        end
    end

    dim_measures = [zeros(Float64, N_dim) for i=1:N_dim_measures]
    global_measures = zeros(Float64, N_dim_global_measures)
    # per dimension measures
    for i_dim in state_filter
        sol_i = sol[i_dim,2:end]
        if cyclic_setback
            _cyclic_setback!(sol_i)
        end

        if flag_past_measures
            for i_meas=1:N_dim_measures
                # collect previous measures
                past_measures = zeros(i_meas-1)
                for j_meas=1:(i_meas-1)
                    past_measures[j_meas] = dim_measures[j_meas][i_dim]
                end
                dim_measures[i_meas][i_dim] = eval_funcs[i_meas](sol_i, past_measures)
            end
        else
            for i_meas=1:N_dim_measures
                dim_measures[i_meas][i_dim] = eval_funcs[i_meas](sol_i)
            end
        end
    end

    # measures using all dimensions
    for i_meas=1:N_dim_global_measures
        global_measures[i_meas] = global_eval_funcs[i_meas](sol[:,2:end])
    end
    (tuple(dim_measures...,global_measures...),false)
end

# MonteCarloProblem needs a function with only (sol, i) as inputs and this way the default of all dimensions beeing evaluated is easier to handle than with an optional/keyword argument
function meval_ode_run(sol, i)
    N_dim = length(sol.prob.u0)
    state_filter = collect(1:N_dim)
    meanval(u::AbstractArray, past_measures::AbstractArray) = StatsBase.mean(u)
    standarddev(u::AbstractArray, past_measures::AbstractArray) = StatsBase.std(u; mean=past_measures[1], corrected=true)
    eval_funcs = [meanval, standarddev, empirical_1D_KL_divergence_hist]
    global_eval_funcs = []
    eval_ode_run(sol, i, state_filter, eval_funcs, global_eval_funcs; flag_past_measures=true)
end


ko_emcp = DEMCBBProblem(rp, ic_ranges, pars, (:K, k_range), eval_ode_run, tail_frac)
ko_sol = solve(ko_emcp)

# random + range
ic_ranges = ()->rand(ic_dist)
k_range = 1.:0.5:3.
N_ics = 20

ko_emcp = DEMCBBProblem(rp, ic_ranges, N_ics, pars, (:K, k_range), my_eval_ode_run, tail_frac)
ko_sol = solve(ko_emcp)

# define a random array
ic_array = ()->rand(ic_dist, N)
k_range = ()->rand(kdist)
ko_emcp = DEMCBBProblem(rp, ic_ranges, N_ics, pars, (:K, k_range), eval_ode_run, tail_frac)
ko_sol = solve(ko_emcp)




# random + random
ic_ranges = [()->rand(ic_dist)]
k_range = (i)->rand(kdist)

ko_emcp = DEMCBBProblem(rp, ic_ranges, N_ics, pars, (:K, k_range), eval_ode_run, tail_frac)
ko_sol = solve(ko_emcp)

D = distance_matrix(ko_sol, ko_emcp, [1.,0.5,0.5,1], histograms=true);

D = distance_matrix(ko_sol, ko_emcp, [1.,0.5,0.5,1.]);
k = 4
fdist = k_dist(D,k);



# analysis
db_eps = 1
db_res = dbscan(D,db_eps,k)
cluster_meas = cluster_means(ko_sol,db_res);
cluster_n = cluster_n_noise(db_res);
cluster_members = cluster_membership(ko_emcp,db_res,0.2,0.05);
(p_win, cluster_measures_dim, cluster_measures_global) = cluster_measures(ko_emcp, ko_sol, db_res, 0.2, 0.05);
cluster_measures_sliding_histograms(ko_emcp, ko_sol, db_res, 1, 0.2, 0.05);
cisc = ClusterICSpaces(ko_emcp, ko_sol, db_res)

true
