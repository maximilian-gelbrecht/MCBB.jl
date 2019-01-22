###########
##### Results Evaluation functions
###########
using Distributions, Clustering, StatsBase
#using PairwiseListMatrices

"""
     distance_matrix(sol::myMCSol, prob::myMCProblem, distance_func::Function, weights::AbstractArray; relative_parameter::Bool=false)

Calculate the distance matrix between all individual solutions.

* `sol`: solution
* `prob`: problem
* `distance_func`: The actual calculating the distance between the measures/parameters of each solution with each other. Signature should be `(measure_1::Union{Array,Number}, measure_2::Union{Array,Number}) -> distance::Number. Example and default is `(x,y)->sum(abs.(x .- y))`.
* `weights`: Instead of the actual measure `weights[i_measure]*measure` is handed over to `distance_func`. Thus `weights` need to be ``N_{meas}+N_{par}`` long array.
* `relative_parameter`: If true, the paramater values during distance calcuation is rescaled to [0,1]

Note, there are also deprecated, other versions of distance_matrix in the code.
"""
function distance_matrix(sol::myMCSol, prob::myMCProblem, distance_func::Function, weights::AbstractArray; relative_parameter::Bool=false)
    mat_elements = zeros((sol.N_mc, sol.N_mc))

    pars = parameter(prob)
    N_pars = length(ParameterVar(prob))
    par_c = copy(pars)

    if relative_parameter
        for i_par=1:N_pars
            min_par = minimum(par[:,i_par])
            max_par = maximum(par[:,i_par])
            #par_range = max_par - min_par
            par_rel = (par[:,i_par] .- min_par)./max_par
            par_c[:,i_par] = par_rel
        end
    end

    for i=1:sol.N_mc
        for j=i+1:sol.N_mc
            for i_meas=1:sol.N_meas
                mat_elements[i,j] += distance_func(weights[i_meas]*sol.sol[i][i_meas], weights[i_meas]*sol.sol[j][i_meas])
            end
            for i_par=1:N_pars
                mat_elements[i,j] += distance_func(weights[sol.N_meas+i_par]*par_c[i,i_par], weights[sol.N_meas+i_par]*par_c[j,i_par])
            end
        end
    end
    mat_elements += transpose(mat_elements)
end
distance_matrix(sol::myMCSol, prob::myMCProblem, weights::AbstractArray; relative_parameter::Bool=false) = distance_matrix(sol, prob, (x,y)->sum(abs.(x .- y)), weights)


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

"""
    cluster_means(sol::myMCSol, clusters::DbscanResult)

Returns the mean of each measure for each cluster.
"""
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

"""
     cluster_measures(prob::myMCProblem, sol::myMCSol, clusters::DbscanResult, window_size::AbstractArray, window_offset::AbstractArray)
     cluster_measures(prob::myMCProblem, sol::myMCSol, clusters::DbscanResult, window_size::Number, window_offset::Number)

Calculated the measures for each cluster along a sliding window. Can also handle multiple parameters being varied.

* `prob`: problem
* `sol`: solution of `prob`
* `clusters`: results from a DBSCAN run.
* `window_size`: Size of the window. In case multiple paramaters being varied has to be an array.
* `window_offset`: Offset of the sliding window. In case multiple paramaters being varied has to be an array.

Returns a tuple with:
* `parameter_windows`: the center value of the sliding windows, in case multiple parameter are being varied, it is a meshgrid.
* `cluster_measures`: (per dimension) measures on the parameter grid
* `cluster_measures_global`: global measures on the parameter grid
"""
cluster_measures(prob::myMCProblem, sol::myMCSol, clusters::DbscanResult, window_size::Number, window_offset::Number) = cluster_measures(prob, sol, clusters, [window_size], [window_offset])
function cluster_measures(prob::myMCProblem, sol::myMCSol, clusters::DbscanResult, window_size::AbstractArray, window_offset::AbstractArray)

    N_dim = sol.N_dim
    N_meas_dim = sol.N_meas_dim
    N_meas_global = sol.N_meas_global
    N_cluster = length(clusters.seeds) + 1  # plus 1 -> plus "noise cluster" / not clustered points
    N_windows, __ = _sliding_window_parameter(prob, window_size, window_offset)
    N_par = length(N_windows)

    cluster_measures = zeros([N_cluster; N_meas_dim; N_dim; N_windows]...)
    cluster_measures_global = zeros([N_cluster; N_meas_global; N_windows]...)
    p_windows = []

    for i_meas=1:N_meas_dim
        p_windows, cluster_measures[:,i_meas,:,[Colon() for i=1:N_par]...] =  measure_on_parameter_sliding_window(prob, sol, i_meas, clusters, window_size, window_offset)
    end
    for i_meas=N_meas_dim+1:N_meas_dim+N_meas_global
        p_windows, c_temp = measure_on_parameter_sliding_window(prob, sol, i_meas, clusters, window_size, window_offset)
        cluster_measures_global[:,i_meas - N_meas_dim,[Colon() for i=1:N_par]...] =  c_temp[:,1,[Colon() for i=1:N_par]...]
    end
    (p_windows, cluster_measures, cluster_measures_global)
end

"""
    ClusterICSpaces

This function/struct returns the distributions as histograms of ICs (and Parameter) in each dimension for cluster seperatly, it also returns the data itself, means and stds. If additional keyword arguments min_par, max_par are given, it limits the analysis to the specified parameter range.

Fields of the struct:
* `data`: array of array of arrays, the ICs and pars for each cluster and dimension
* `histograms`: N_cluster x N_dim Array of Histograms of ICs/Par
* `means`: Means of each dimension for each cluster
* `stds`: Stds of each dimension for each cluster
* `cross_dim_means`: list of Means of ICs across IC-dimensions per Cluster
* `cross_dim_stds`: list of Std of ICs across IC-dimensions per Cluster
* `cross_dim_kurts`: list of Kurtosis of ICs across IC-dimensions per Cluster

# Constructor

    ClusterICSpaces(prob::myMCProblem, sol::myMCSol, clusters::DbscanResult; min_par::Number=-Inf, max_par::Number=Inf, nbins::Int64=20)

* `prob`: Problem
* `sol`: solution of `prob`
* `clusters`: DBSCAN results
* `min_par`, `max_par`: restrict the analysis to parameters within this value range
* `nbins`: Number of bins of the histograms
"""
struct ClusterICSpaces
    data::AbstractArray
    histograms::AbstractArray
    means::AbstractArray
    stds::AbstractArray

    cross_dim_means::AbstractArray
    cross_dim_stds::AbstractArray
    cross_dim_skews::AbstractArray

    function ClusterICSpaces(prob::myMCProblem, sol::myMCSol, clusters::DbscanResult; min_par::Number=-Inf, max_par::Number=Inf, nbins::Int64=20)

        if length(ParameterVar(prob))>1
            error("Not yet implemented for systems with more than one parameter")
        end

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

            if isempty(data[i_cluster][1])
                continue
            end

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

"""
    cluster_n_noise(clusters::DbscanResult)

Returns the number of points assignt to the "noise" cluster (somehow this is not automaticlly returned by the routine of Clustering.jl).
"""
function cluster_n_noise(clusters::DbscanResult)
    count = 0
    for i=1:length(clusters.assignments)
        if clusters.assignments[i]==0
            count += 1
        end
    end
    count
end

"""
    cluster_membership(par::AbstractArray, clusters::DbscanResult)

Calculates the proportion of members for each cluster for all parameter values.
"""
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

### OLD VERSION, WILL BE DELETED AT SOME POINT
# counts how many solutions are part of the individual clusters for each parameter step
# -  par needs to be 1d mapping each run to its parameter value, e.g. ic_par[:,end]
# this method uses a sliding window over the parameter axis.
# should be used when parameters are randomly generated.
# - normalize: normalize by number of parameters per window
function cluster_membership(par::AbstractArray, clusters::DbscanResult, window_size::Number, window_offset::Number, normalize::Bool=true)
    error("This used to be an old version of cluster_membership, please use cluster_membership(prob::myMCProb, clusters::DbscanResult, window_size, window_offset) now")
end

"""
    cluster_membership(prob::myMCProblem, clusters::DbscanResult, window_size::AbstractArray, window_offset::AbstractArray, normalize::Bool=true)
    cluster_membership(prob::myMCProblem, clusters::DbscanResult, window_size::Number, window_offset::Number, normalize::Bool=true)

Calculates the proportion of members for each cluster within a parameter sliding window.

* `prob`: problem
* `sol`: solution of `prob`
* `clusters`: results from a DBSCAN run.
* `window_size`: Size of the window. In case multiple paramaters being varied has to be an array.
* `window_offset`: Offset of the sliding window. In case multiple paramaters being varied has to be an array.

Returns a tuple with:
* `parameter_windows`: the center value of the sliding windows, in case multiple parameter are being varied, it is a meshgrid.
* `cluster_measures`: members of the clusters on the parameter grid
"""
function cluster_membership(prob::myMCProblem, clusters::DbscanResult, window_size::AbstractArray, window_offset::AbstractArray, normalize::Bool=true)

    N_cluster = length(clusters.seeds) + 1  # plus 1 -> plus "noise cluster" / not clustered points
    ca = clusters.assignments
    N = length(ca)
    N_par = length(ParameterVar(prob))

    N_windows, windows_mins = _sliding_window_parameter(prob, window_size, window_offset)

    memberships = zeros([N_windows;N_cluster]...)
    parameter_mesh = zeros([N_windows;N_par]...)

    for (ip,ic) in zip(Iterators.product(windows_mins...),CartesianIndices(zeros(Int,N_windows...)))

        par_ind = ones(Bool, prob.N_mc)
        for i_par in 1:N_par
            par_ind = par_ind .& ((parameter(prob,i_par) .> ip[i_par]) .& (parameter(prob,i_par) .< (ip[i_par] + window_size[i_par])))
        end

        window_ca = ca[par_ind]
        #println(window_ca)
        N_c_i = 0
        for i_ca in eachindex(window_ca)
            memberships[ic,window_ca[i_ca]+1] += 1
            N_c_i += 1
        end
        for i_cluster=1:N_cluster
            if normalize
                memberships[ic,i_cluster] ./= N_c_i
            end
        end

        parameter_mesh[ic,:] = collect(ip)
    end

    # return
    (parameter_mesh, memberships)
end
cluster_membership(prob::myMCProblem, clusters::DbscanResult, window_size::Number, window_offset::Number, normalize::Bool=true) = cluster_membership(prob, clusters, [window_size], [window_offset], normalize)

"""
    measure_on_parameter_sliding_window

Does calculate measures (per cluster) on parameter sliding windows. This routine is called by `cluster_membership` and `cluster_measures` but can also be used for plotting measures on the parameter grid manually.

_ATTENTION_: If a cluster has no members within a window the value is set to `NaN`. This should simply omit these points from beeing plotted (while `missing` and `nothing` are currently not compatible with most plotting packages).

    measure_on_parameter_sliding_window(prob::myMCProblem, sol::myMCSol, i::Int, clusters::DbscanResult, window_size::Number, window_offset::Number)

Does return the `i`-th measure for each cluster seperatly on the parameter sliding window grid

* `prob`: Problem
* `sol`: solution of `prob`
* `i`: function returns the `i`-th measure
* `clusters`: results from a DBSCAN run.
* `window_size`: Size of the window. In case multiple paramaters being varied has to be an array.
* `window_offset`: Offset of the sliding window. In case multiple paramaters being varied has to be an array.

    measure_on_parameter_sliding_window(prob::myMCProblem, sol::myMCSol, i::Int, window_size::Number, window_offset::Number)

Does return the `i`-th measure on the parameter sliding window grid (does _not_ calculate the measure for each cluster seperatly)

All methods return a tuple with:
* `parameter_windows`: the center value of the sliding windows, in case multiple parameter are being varied, it is a meshgrid.
* `cluster_measures`: members of the clusters on the parameter grid
"""
measure_on_parameter_sliding_window(prob::myMCProblem, sol::myMCSol, i::Int, window_size::Number, window_offset::Number) = measure_on_parameter_sliding_window(prob, sol, i, [window_size], [window_offset])
measure_on_parameter_sliding_window(prob::myMCProblem, sol::myMCSol, i::Int, clusters::DbscanResult, window_size::Number, window_offset::Number) = measure_on_parameter_sliding_window(prob, sol, i, clusters, [window_size], [window_offset])
measure_on_parameter_sliding_window(prob::myMCProblem, sol::myMCSol, i::Int, window_size::AbstractArray, window_offset::AbstractArray) = measure_on_parameter_sliding_window(prob, sol, i, zeros(Int,prob.N_mc), 1, window_size, window_offset)
measure_on_parameter_sliding_window(prob::myMCProblem, sol::myMCSol, i::Int, clusters::DbscanResult, window_size::AbstractArray, window_offset::AbstractArray) = measure_on_parameter_sliding_window(prob, sol, i, clusters.assignments, length(clusters.seeds) + 1, window_size, window_offset)
function measure_on_parameter_sliding_window(prob::myMCProblem, sol::myMCSol, i::Int, clusters_assignments::AbstractArray, N_cluster::Int, window_size::AbstractArray, window_offset::AbstractArray)

    ca = clusters_assignments
    N = length(ca)
    N_par = length(ParameterVar(prob))

    N_windows, windows_mins = _sliding_window_parameter(prob, window_size, window_offset)

    # get dimension of measure
    obs = get_measure(sol, i)
    if length(size(obs))==1
        N_dim = 1
    else
        __, N_dim = size(obs)
    end

    #cluster_meas = zeros([N_windows;N_dim;N_cluster]...)
    cluster_meas = zeros([N_cluster;N_dim;N_windows]...)
    parameter_mesh = zeros([N_par;N_windows]...)

    for (ip,ic) in zip(Iterators.product(windows_mins...),CartesianIndices(zeros(Int,N_windows...)))

        par_ind = ones(Bool, prob.N_mc)
        for i_par in 1:N_par
            par_ind = par_ind .& ((parameter(prob,i_par) .> ip[i_par]) .& (parameter(prob,i_par) .< (ip[i_par] + window_size[i_par])))
        end

        window_ca = ca[par_ind]
        par_ind_numbers = findall(par_ind)
        # here we need to do something
        N_c_i = zeros(Int,(N_dim, N_cluster)) # counts how many values are within the window for each cluster (for normalization)

        # collect and copy data
        for i_ca in eachindex(window_ca)
            for i_dim=1:N_dim
                cluster_meas[window_ca[i_ca]+1, i_dim, ic] += sol.sol[par_ind_numbers[i_ca]][i][i_dim]
                N_c_i[i_dim, window_ca[i_ca]+1] += 1
            end
        end

        # normalize/average it
        for i_cluster=1:N_cluster
            for i_dim=1:N_dim
                if !(N_c_i[i_dim,i_cluster] == 0)
                    cluster_meas[i_cluster, i_dim, ic] /= N_c_i[i_dim, i_cluster]
                else
                    cluster_meas[i_cluster, i_dim, ic] = NaN
                end

            end
        end

        parameter_mesh[:,ic] = collect(ip)
    end
    if N_par == 1
        parameter_mesh = parameter_mesh[1,:]
    end
    (parameter_mesh, cluster_meas)
end

# function to calculate the sliding window parameter Array
_sliding_window_parameter(prob::myMCProblem, window_size::Number, window_offset::Number) = _sliding_window_parameter(prob, [window_size], [window_offset])
function _sliding_window_parameter(prob::myMCProblem, window_size::AbstractArray, window_offset::AbstractArray)

    N_par = length(ParameterVar(prob))
    if (length(window_size)!=N_par)|(length(window_offset)!=N_par)
        error("Window Size and Window Offset need to have as many elements as they are parameters")
    end

    N_windows = zeros(Int,N_par)
    windows_mins = []

    # go over every parameter
    for i_par = 1:N_par
        min_par = minimum(parameter(prob,i_par))
        max_par = maximum(parameter(prob,i_par))

        push!(windows_mins, min_par:window_offset[i_par]:(max_par-window_size[i_par]))
        N_windows[i_par] = length(windows_mins[i_par])

        if N_windows[i_par] <= 1
            @warn "Only 1 or less Windows in cluster_membership"
        end
    end
    (N_windows, windows_mins)
end

"""
     k_dist(D::AbstractArray, k::Int=4)

Helper function for estimating a espilon value for DBSCAN. In the original paper, Ester et al. suggest to plot the `k`-dist graph (espacially for ``k=4``) to estimate a value for `eps` given ``minPts = k``. It computes the distance to the `k`-th nearast neighbour for all data points given their distance matrix.

* `D`: Distance matrix
* `k`: calculate the distance to the `k`-th neighbour

Returns sorted array with the k-dist of all elements of `D`.
"""
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

"""
    KNN_dist_relative(D::AbstractArray, rel_K::Float64=0.005)

Returns the cumulative distance to the `rel_K*N` nearest neighbour.

* `D`: Distance matrix
* `rel_K`
"""
function KNN_dist_relative(D::AbstractArray, rel_K::Float64=0.005)
    (N, N_2) = size(D)
    K = Int(round(N * rel_K))
    KNN_dist(D, K)
end

"""
    KNN_dist(D::AbstractArray, K::Int)

Returns the cumulative `K-`th nearest neighbour distance.

* `D`: Distance matrix
* `K`
"""
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
