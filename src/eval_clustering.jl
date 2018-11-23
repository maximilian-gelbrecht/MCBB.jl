###########
##### Results Evaluation functions
###########
using Distributions, Clustering, StatsBase
#using PairwiseListMatrices

# calculates the distance matrix
# also incorporates the parameter values as additional weights
#
# sol: solution
# par: parameter as Array of length sol.N_mc
# distance_func: distance function that maps (x1,x2,p1,p2) -> D(x1,x2; p1,p2)
# relative parameter: if true the parameter values are rescaled to be within [0,1]
function distance_matrix(sol::myMCSol, par::AbstractArray, distance_func::Function, relative_parameter::Bool=false)
    mat_elements = zeros((sol.N_mc,sol.N_mc))
    if relative_parameter
        min_par = minimum(par)
        max_par = maximum(par)
        #par_range = max_par - min_par
        par_rel = (par .- min_par)./max_par
        par_c = par_rel
    else
        par_c = copy(par) # do we really need the copy here? I don't know
    end

    """
    # PairwiseListMatrix is not supported for 1.0 yet

    N_entries = (sol.N_mc * (sol.N_mc - 1)) / 2

    i_tot = 0
    for i=1:sol.N_mc
        for j=i+1:sol.N_mc
            i_tot += 1
            mat_elements[i_tot] = distance_func(sol.sol[i], sol.sol[j], par[i], par[j])
        end
    end
    PairwiseListMatrix(mat_elements)
    """

    for i=1:sol.N_mc
        for j=i+1:sol.N_mc
            mat_elements[i,j] = distance_func(sol.sol[i], sol.sol[j], par_c[i], par_c[j])

        end
    end
    mat_elements += transpose(mat_elements)
end

# more handy version with distance function already defined
function distance_matrix(sol::myMCSol, par::AbstractArray, weights::AbstractArray=[1.,0.5,0.5,1.], relative_parameter::Bool=false)
    if length(weights)!=(length(sol.sol.u[1])+1) # +1 because of the parameter
        error("Length of weights does not fit length of solution measurements")
    end
    distance_matrix(sol, par, (x,y,p1,p2) -> weighted_norm(x,y,p1,p2,weights), relative_parameter)
end

function distance_matrix(sol::myMCSol, distance_func::Function)
    #N_entries = (sol.N_mc * (sol.N_mc - 1)) / 2
    #mat_elements = zeros(N_entries)
    mat_elements = zeros((sol.N_mc,sol.N_mc))
    # add parameter to solutions
    """
    # PairwiseListMatrix is not supported for 1.0 yet
    i_tot = 0
    for i=1:sol.N_mc
        for j=i+1:sol.N_mc
            i_tot += 1
            mat_elements[i_tot] = distance_func(sol.sol[i],sol.sol[j])
        end
    end
    PairwiseListMatrix(mat_elements)
    """
    for i=1:sol.N_mc
        for j=i+1:sol.N_mc
            mat_elements[i,j] = distance_func(sol.sol[i],sol.sol[j])
        end
    end
    mat_elements += transpose(mat_elements)
end
distance_matrix(sol::myMCSol) = distance_matrix(sol, weighted_norm)


# calculated the weighted norm between two trajectories, so one entry of the distance matrix
# x, y :: Tuples or Arrays containing all measures of the trajectories (e.g. means, vars per spatial dimension)
#
function weighted_norm(x, y, norm_function::Function, weights::AbstractArray=[1., 0.5, 0.5])
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
function cluster_means(sol::myMCSol, clusters::DbscanResult)
    N_cluster = length(clusters.seeds)+1 # plus 1 -> plus "noise cluster" / not clustered points
    N_dim = sol.N_dim
    mean_measures = zeros((N_cluster,sol.N_meas_dim,N_dim))
    for i_sol=1:sol.N_mc
        for i_meas=1:sol.N_meas_dim
            mean_measures[clusters.assignments[i_sol]+1,i_meas,:] += sol.sol[i_sol][i_meas]
        end
    end
    mean_measures ./ sol.N_mc
end

# returns for each cluster seperatly per dimension and per measure the (parameter, value of measure) pairs
# this is accumulated (and normalized) over a sliding window
#
# returns the array with the parameter values of each window and a 4-dimensional matrix with dimensions:
#           - i_cluster: number of cluster (noise cluster is the first cluster)
#           - i_meas: number of the measures (e.g. 1 is usually mean, 2 is std and so on)
#           - i_dim: number of the dimension (of the system)
#           - i_window: number of the window/parameter value
#
function cluster_measures(prob::myMCProblem, sol::myMCSol, clusters::DbscanResult, window_size::Number, window_offset::Number)
    N_cluster = length(clusters.seeds)+1 # plus 1 -> plus "noise cluster" / not clustered points
    N_dim = sol.N_dim
    par = parameter(prob)
    ca = clusters.assignments

    # windows
    min_par = minimum(par)
    max_par = maximum(par)
    par_range = max_par - min_par

    N_windows = Int(ceil(par_range/window_offset)) - Int(ceil(window_size/window_offset)) + 1
    if N_windows <= 1
        warn("Only 1 or less Windows in cluster_measures")
    end
    p_windows = zeros(N_windows)
    cluster_measures = zeros((N_cluster, sol.N_meas_dim, N_dim, N_windows))
    cluster_measures_global = zeros((N_cluster, sol.N_meas_global, N_windows))

    for i=1:N_windows
        window_min = min_par + (i-1)*window_offset
        window_max = window_min + window_size
        p_windows[i] = 0.5*(window_min + window_max)

        par_ind_bool = (par .>= window_min) .& (par .< window_max) # parameter indices as bools
        par_ind = findall(par_ind_bool) # parameter indices as numbers

        window_ca = ca[par_ind] # cluster assigments in this window

        N_c_i = zeros(Int,(N_cluster, sol.N_meas_dim, N_dim)) # counts how many values are within the window for each cluster (for normalization)
        N_c_i_global = zeros(Int, (N_cluster, sol.N_meas_global))

        # collect and copy data
        for i_ca in eachindex(window_ca)
            for i_meas=1:sol.N_meas_dim
                for i_dim=1:N_dim
                    cluster_measures[window_ca[i_ca]+1, i_meas, i_dim, i] += sol.sol[par_ind[i_ca]][i_meas][i_dim]
                    N_c_i[window_ca[i_ca]+1, i_meas, i_dim] += 1
                end
            end
            for i_meas=sol.N_meas_dim+1:sol.N_meas
                cluster_measures_global[window_ca[i_ca]+1, i_meas - sol.N_meas_dim, i] += sol.sol[par_ind[i_ca]][i_meas] # correct i-meas index
                N_c_i_global[window_ca[i_ca]+1, i_meas - sol.N_meas_dim] += 1
            end
        end

        # normalize/average it
        for i_cluster=1:N_cluster
            for i_meas=1:sol.N_meas_dim
                for i_dim=1:N_dim
                    if !(N_c_i[i_cluster, i_meas, i_dim] == 0)
                        cluster_measures[i_cluster, i_meas, i_dim, i] /= N_c_i[i_cluster, i_meas, i_dim]
                    end
                end
            end
            for i_meas=1:sol.N_meas_global
                cluster_measures_global[i_cluster, i_meas, i] /= N_c_i_global[i_cluster, i_meas]
            end
        end
    end
    (p_windows, cluster_measures, cluster_measures_global)
end

# This function/struct returns the distributions as histograms of ICs (and Parameter) in each dimension for cluster seperatly, it also returns the data itself, means and stds. If additional keyword arguments min_par, max_par are given, it limits the analysis to the specified parameter range
# fields of the struct:
#               - data : array of array of arrays, the ICs and pars for each cluster and dimension
#               - histograms: N_cluster x N_dim Array of Histograms of ICs/Par
#               - means: Means of each dimension for each cluster
#               - stds: Stds of each dimension for each cluster
#
#               - cross_dim_means: list of Means of ICs across IC-dimensions per Cluster
#               - cross_dim_stds: list of Std of ICs across IC-dimensions per Cluster
#               - cross_dim_kurts: list of Kurtosis of ICs across IC-dimensions per Cluster
#
struct ClusterICSpaces
    data::AbstractArray
    histograms::AbstractArray
    means::AbstractArray
    stds::AbstractArray

    cross_dim_means::AbstractArray
    cross_dim_stds::AbstractArray
    cross_dim_skews::AbstractArray

    function ClusterICSpaces(prob::myMCProblem, sol::myMCSol, clusters::DbscanResult; min_par::Number=-Inf, max_par::Number=Inf, nbins::Int64=20)

        N_cluster = length(clusters.seeds)+1 # plus 1 -> plus "noise cluster" / not clustered points
        N_dim = sol.N_dim

        icp = prob.ic_par
        ca = clusters.assignments

        # collect the data for each cluster and dimension
        cross_dim_means = [[] for i=1:N_cluster]
        cross_dim_stds = [[] for i=1:N_cluster]
        cross_dim_skews = [[] for i=1:N_cluster]

        data = [[[] for i=1:N_dim+1] for i=1:N_cluster] # +1 for the parameter
        for i=1:sol.N_mc
            if (icp[i,end] > min_par) & (icp[i,end] < max_par)
                i_cluster = ca[i] + 1  # plus 1 -> plus "noise cluster" / not clustered points
                for i_dim=1:N_dim # ICs
                    push!(data[i_cluster][i_dim],icp[i,i_dim])
                end

                i_mean, i_std = mean_and_std(icp[i,1:N_dim])
                i_skew = skewness(icp[i,1:N_dim], i_mean)
                push!(cross_dim_means[i_cluster], i_mean)
                push!(cross_dim_stds[i_cluster], i_std)
                push!(cross_dim_skews[i_cluster], i_skew)
                push!(data[i_cluster][N_dim+1],icp[i,N_dim+1]) # parameter
            end
        end

        for i_cluster=1:N_cluster
            for i_dim=1:N_dim+1
                data[i_cluster][i_dim] = convert.(Float64,data[i_cluster][i_dim])
            end
        end
        data = convert(Array{Array{Array{Float64,1},1},1}, data)
        cross_dim_means = convert(Array{Array{Float64,1},1}, cross_dim_means)
        cross_dim_stds = convert(Array{Array{Float64,1},1}, cross_dim_stds)
        cross_dim_skews = convert(Array{Array{Float64,1},1}, cross_dim_skews)

        # fit histograms
        # the automatic bin edges of julia are somewhat lackluster and unconsistent, we define our own


        hists = [[] for i=1:N_cluster]
        for i_cluster=1:N_cluster
            ic_min = minimum(minimum(data[i_cluster]))
            ic_max = maximum(maximum(data[i_cluster]))
            ic_range = ic_max - ic_min

            edges = ic_min:ic_range/nbins:ic_max
            for i_dim=1:N_dim+1
                push!(hists[i_cluster],normalize(fit(Histogram, data[i_cluster][i_dim], edges, closed=:left)))
            end
        end

        # compute means and std
        means = zeros(N_cluster,N_dim+1)
        stds = zeros(N_cluster,N_dim+1)
        for i_cluster=1:N_cluster
            for i_dim=1:N_dim+1
                (means[i_cluster,i_dim], stds[i_cluster,i_dim]) = mean_and_std(data[i_cluster][i_dim])
            end
        end

        new(data, hists, means, stds, cross_dim_means, cross_dim_stds, cross_dim_skews)
    end
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
            if !(N_c_i == 0)
                memberships[i, :] /= N_c_i
            end
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

function KNN_dist_relative(D::AbstractArray, rel_K::Float64=0.005)
    (N, N_2) = size(D)
    K = Int(round(N * rel_K))
    KNN_dist(D, K)
end

function KNN_dist(D::AbstractArray, K::Int)
    (N, N_2) = size(D)
    if N!=N_2
        error("k_dist: Input Matrix has to be a square matrix")
    end
    k_d = zeros(N, K)
    for i=1:N
        D_i_s = sort(D[i,:])
        for ik=1:K
            k_d[i,ik] = D_i_s[ik]
        end
    end
    sum(k_d, dims=2) #? STIMMT DAS?
end
