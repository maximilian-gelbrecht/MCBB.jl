###########
##### Results Evaluation functions
###########
using Distributions, Clustering

# compute the distance matrix used for the dbSCAN clustering. here, we could experiment with different ways how to setup this matrix
# OLD VERSION: returns a dense matrix instead of a PairwiseListMatrix
function distance_matrix_dense(sol::myMCSol, distance_func::Function)
    D = zeros((sol.N_mc, sol.N_mc))
    for i=1:sol.N_mc
        for j=1:i
            D[i,j] = distance_func(sol.sol.u[i], sol.sol.u[j])
        end
    end

    for i=1:sol.N_mc
        for j=i:sol.N_mc
            D[i,j] = D[j,i]
        end
    end
    D
end
distance_matrix_dense(sol::myMCSol) = distance_matrix_dense(sol, weighted_norm)

# also includes the parameters into the distances calculation, thus favoring pairs that have similar parameters. uses the relative parameter distance. needs parameter from combined ic-par matrix as input, so that length(par)==N_mc
function distance_matrix_dense(sol::myMCSol, par::AbstractArray, distance_func::Function)
    # transform parameter vector to measure the relative distances
    min_par = minimum(par)
    max_par = maximum(par)
    par_range = max_par - min_par
    par_rel = (par .- min_par)./par_range

    D = zeros((sol.N_mc, sol.N_mc))
    for i=1:sol.N_mc
        xi = tuple(sol.sol.u[i]..., par_rel[i]) # deepcopy cause the push! would also modify the original sol object
        for j=1:i
            xj = tuple(sol.sol.u[j]..., par_rel[j])
            D[i,j] = distance_func(xi,xj)
        end
    end

    for i=1:sol.N_mc
        for j=i:sol.N_mc
            D[i,j] = D[j,i]
        end
    end
    D
end
distance_matrix_dense(sol::myMCSol, par::AbstractArray) = distance_matrix_dense(sol, par, (x,y) -> weighted_norm(x,y,[1.,0.5,0.5,0.25,1]))

# test distance matrix to output pairwisematrix to save memory.
# also incorporates the parameter values as additional weights
function distance_matrix(sol::myMCSol, par::AbstractArray, distance_func::Function)
    min_par = minimum(par)
    max_par = maximum(par)
    par_range = max_par - min_par
    par_rel = (par .- min_par)./par_range
    N_entries = (sol.N_mc * (sol.N_mc - 1)) / 2
    mat_elements = zeros(N_entries)

    # add parameter to solutions
    i_tot = 0
    for i=1:sol.N_mc
        #xi = tuple(sol.sol.u[i]..., par_rel[i])
        for j=i+1:sol.N_mc
            i_tot += 1
            #xj = tuple(sol.sol.u[j]..., par_rel[j])
            #mat_elements[i_tot] = distance_func(xi,xj)
            mat_elements[i_tot] = distance_func(sol.sol.u[i], sol.sol.u[j], par[i], par[j])
        end
    end

    PairwiseListMatrix(mat_elements)
end
distance_matrix(sol::myMCSol, par::AbstractArray) = distance_matrix(sol, par, (x,y,p1,p2) -> weighted_norm(x,y,p1,p2,[1,0.5,0.5,0.25,1]))

function distance_matrix(sol::myMCSol, distance_func::Function)
    N_entries = (sol.N_mc * (sol.N_mc - 1)) / 2
    mat_elements = zeros(N_entries)
    # add parameter to solutions
    i_tot = 0
    for i=1:sol.N_mc
        for j=i+1:sol.N_mc
            i_tot += 1
            mat_elements[i_tot] = distance_func(sol.sol.u[i],sol.sol.u[j])
        end
    end
    PairwiseListMatrix(mat_elements)
end
distance_matrix(sol::myMCSol) = distance_matrix(sol, weighted_norm)

# calculated the weighted norm between two trajectories, so one entry of the distance matrix
# x, y :: Tuples or Arrays containing all measures of the trajectories (e.g. means, vars per spatial dimension)
#
function weighted_norm(x, y, norm_function::Function, weights::AbstractArray=[1., 0.5, 0.5, 0.25])
    N_dim_meas::Int64 = length(x)
    out::Float64 = 0.
    for i_dim=1:N_dim_meas
        out += weights[i_dim]*norm_function(x[i_dim],y[i_dim])
    end
    out
end
# use l1-norm by default
weighted_norm(x, y, weights::AbstractArray=[1., 0.5, 0.5, 0.25]) = weighted_norm(x,y,(x,y) -> sum(abs.(x .- y)), weights)

# weighted norm that also weights in the parameter values
function weighted_norm(x, y, par1::Number, par2::Number, norm_function::Function, weights::AbstractArray=[1, 0.5, 0.5, 0.25, 1])
    weighted_norm(x,y, norm_function, weights[1:(end-1)]) + weights[end]*norm_function(par1,par2)
end
weighted_norm(x,y,par1::Number,par2::Number, weights::AbstractArray=[1., 0.5, 0.5, 0.25, 1.]) = weighted_norm(x,y,par1,par2,(x,y) -> sum(abs.(x .- y)), weights)

function cluster()
    false
end

# return mean values of all measures for each cluster
function cluster_measures(sol::myMCSol, clusters::DbscanResult)
    N_cluster = length(clusters.seeds)+1 # plus 1 -> plus "noise cluster" / not clustered points
    N_dim = length(sol.sol.u[1][1])
    mean_measures = zeros((N_cluster,sol.N_meas,N_dim))
    for i_sol=1:sol.N_mc
        for i_meas=1:sol.N_meas
            mean_measures[clusters.assignments[i_sol]+1,i_meas,:] += sol.sol.u[i_sol][i_meas]
        end
    end
    mean_measures ./ sol.N_mc
end

# returns the number of points assignt to the "noise" cluster (somehow this is not automaticlly returned by the routine)
function cluster_n_noise(clusters::DbscanResult)
    count = 0
    for i=1:length(clusters.assignments)
        if clusters.assignments[i]==0
            count += 1
        end
    end
    count
end

# counts how many solutions are part of the individual clusters for each parameter step
# par needs to be 1d mapping each run to its parameter value, e.g. ic_par[:,end]
function cluster_membership(par::AbstractArray, clusters::DbscanResult)
    N_cluster = length(clusters.seeds) + 1  # plus 1 -> plus "noise cluster" / not clustered points
    ca = clusters.assignments
    N = length(ca)
    par_u = unique(par)
    N_par = length(par_u)

    memberships = zeros(N_par, N_cluster)
    for i=1:N
        i_par = searchsortedfirst(par_u,par[i])
        memberships[i_par, ca[i]+1] += 1
    end
    memberships
end

# counts how many solutions are part of the individual clusters for each parameter step
# -  par needs to be 1d mapping each run to its parameter value, e.g. ic_par[:,end]
# this method uses a sliding window over the parameter axis.
# should be used when parameters are randomly generated.
# - normalize: normalize by number of parameters per window

# NOT PROPERLY TESTET YET,
# ONLY TESTED THAT IT OUTPUTS SOMETHING
function cluster_membership(par::AbstractArray, clusters::DbscanResult, window_size::Number, window_offset::Number, normalize::Bool=true)
    N_cluster = length(clusters.seeds) + 1  # plus 1 -> plus "noise cluster" / not clustered points
    ca = clusters.assignments
    N = length(ca)

    min_par = minimum(par)
    max_par = maximum(par)
    par_range = max_par - min_par

    N_windows = Int(ceil(par_range/window_offset)) - Int(ceil(window_size/window_offset)) + 1
    if N_windows <= 1
        warn("Only 1 or less Windows in cluster_membership")
    end

    p_windows = zeros(N_windows)
    memberships = zeros(N_windows, N_cluster)

    for i=1:N_windows
        window_min = min_par + (i-1)*window_offset
        window_max = window_min + window_size
        p_windows[i] = 0.5*(window_min + window_max)

        par_ind = (par .>= window_min) .& (par .<= window_max)
        window_ca = ca[par_ind]

        N_c_i = 0
        for i_ca in eachindex(window_ca)
            memberships[i, window_ca[i_ca]+1] += 1
            N_c_i += 1
        end
        if normalize
            memberships[i, :] /= N_c_i
        end
    end
    (p_windows, memberships)
end


# helper function for estimating a espilon value for dbscan.
# in the original paper, Ester et al. suggest to plot the k-dist graph (espaccially for k=4) to estimate a value for eps given minPts = k
# it computes the distance to the k-th nearast neighbour for all data points given their distance matrix
function k_dist(D::AbstractArray, k::Int=4)
    (N, N_2) = size(D)
    if N!=N_2
        error("k_dist: Input Matrix has to be a square matrix")
    end
    k_d = zeros(N)
    # calculate k-dist for each point
    for i=1:N
        D_i_s = sort(D[i,:])
        k_d[i] = D_i_s[k]
    end
    sort(k_d, rev=true)
end
