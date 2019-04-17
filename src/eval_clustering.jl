###########
##### Results Evaluation functions
###########
using Distributions, Clustering, StatsBase
#using PairwiseListMatrices

"""
     distance_matrix(sol::myMCSol, prob::myMCProblem, distance_func::Function, weights::AbstractArray; relative_parameter::Bool=false, histograms::Bool=false)

Calculate the distance matrix between all individual solutions.

* `sol`: solution
* `prob`: problem
* `distance_func`: The actual calculating the distance between the measures/parameters of each solution with each other. Signature should be `(measure_1::Union{Array,Number}, measure_2::Union{Array,Number}) -> distance::Number. Example and default is `(x,y)->sum(abs.(x .- y))`.
* `weights`: Instead of the actual measure `weights[i_measure]*measure` is handed over to `distance_func`. Thus `weights` need to be ``N_{meas}+N_{par}`` long array.
* `relative_parameter`: If true, the paramater values during distance calcuation is rescaled to [0,1]
* `histograms::Bool`: If true, the distance calculation is based on [`distance_matrix_histogram`](@ref) with the default histogram distance [`wasserstein_histogram_distance`](@ref).
"""
function distance_matrix(sol::myMCSol, prob::myMCProblem, distance_func::Function, weights::AbstractArray; matrix_distance_func::Union{Function, Nothing}=nothing, relative_parameter::Bool=false, histograms::Bool=false)

    pars = parameter(prob)
    N_pars = length(ParameterVar(prob))
    par_c = copy(pars)

    if (sol.N_meas_matrix!=0) & (matrix_distance_func==nothing)
        error("There is a matrix measure in the solution but no distance func for it.")
    end

    if relative_parameter
        for i_par=1:N_pars
            min_par = minimum(par[:,i_par])
            max_par = maximum(par[:,i_par])
            #par_range = max_par - min_par
            par_rel = (par[:,i_par] .- min_par)./max_par
            par_c[:,i_par] = par_rel
        end
    end

    if histograms
        mat_elements = distance_matrix_histogram(sol, par_c, distance_func, weights; matrix_distance_func=matrix_distance_func)
    else
        mat_elements = zeros((sol.N_mc, sol.N_mc))

        for i=1:sol.N_mc
            for j=i+1:sol.N_mc
                for i_meas=1:sol.N_meas_dim
                    mat_elements[i,j] += weights[i_meas]*distance_func(sol.sol[i][i_meas], sol.sol[j][i_meas])
                end
                for i_meas =sol.N_meas_dim+1:sol.N_meas_dim+sol.N_meas_matrix
                    mat_elements[i,j] += weights[i_meas]*matrix_distance_func(sol.sol[i][i_meas], sol.sol[j][i_meas])
                end
                for i_meas = sol.N_meas_dim+sol.N_meas_matrix+1:sol.N_meas
                    mat_elements[i,j] += weights[i_meas]*distance_func(sol.sol[i][i_meas], sol.sol[j][i_meas])
                end

                for i_par=1:N_pars
                    mat_elements[i,j] += weights[sol.N_meas+i_par]*distance_func(par_c[i,i_par], par_c[j,i_par])
                end
            end
        end
        mat_elements += transpose(mat_elements)
    end

    if sum(isnan.(mat_elements))>0
        @warn "There are some elements NaN in the distance matrix"
    end
    if sum(isinf.(mat_elements))>0
        @warn "There are some elements Inf in the distance matrix"
    end
    mat_elements
end
distance_matrix(sol::myMCSol, prob::myMCProblem, weights::AbstractArray; relative_parameter::Bool=false, histograms::Bool=false, matrix_distance_func::Union{Function, Nothing}=nothing) = distance_matrix(sol, prob, (x,y)->sum(abs.(x .- y)), weights; relative_parameter=relative_parameter, histograms=histograms, matrix_distance_func=matrix_distance_func)

"""
    distance_matrix_histogram(sol::myMCSol, pars::AbstractArray{T}, distance_func, weights::AbstractArray{S}, histogram_distance::Function; matrix_distance_func::Union{Function, Nothing}=nothing, ecdf::Bool=true, k_bin::Int=1)

This function is called by `distance_matrix` if it is called with the `histograms` flag `true`.

Computes for each run in the solution `sol` for each measure a histogram of the measures of all system dimensions. The binning of the histograms is computed with Freedman-Draconis rule and the same across all runs for each measure.

The distance matrix is then computed given a suitable histogram distance function `histogram_distance` between these histograms.

This is intended to be used in order to avoid symmetric configurations in larger systems to be distinguished from each other. Example: Given a system with 10 identical oscillators. Given this distance calculation a state where oscillator 1-5 are synchronized and 6-10 are not syncronized would be in the same cluster as a state where oscillator 6-10 are synchronized and 1-5 are not synchronized. If you don't want this kind of behaviour, use the regular `distance_matrix` function.

Inputs:
* `sol::myMCSol`: solution
* `par::AbstractArray`: parameter array (can also be called with a [`myMCProblem`](@ref) from which the parameters will be automatically taken)
* `distance_func`: The actual calculating the distance between the measures/parameters of each solution with each other. Signature should be `(measure_1::Union{Array,Number}, measure_2::Union{Array,Number}) -> distance::Number. Example and default is `(x,y)->sum(abs.(x .- y))`.
* `weights`: Instead of the actual measure `weights[i_measure]*measure` is handed over to `distance_func`. Thus `weights` need to be ``N_{meas}+N_{par}`` long array.
* `histogram_distance`: The distance function between two histograms. Default is [`wasserstein_histogram_distance`](@ref).
* `matrix_distance_func`: The distance function between two matrices or arrays or length different from ``N_{dim}``. Used e.g. for Crosscorrelation.
* `ecdf::Bool` if true the `histogram_distance` function gets the empirical cdfs instead of the histogram
* `k_bin::Int`: Multiplier to increase (``k_{bin}>1``) or decrease the bin width and thus decrease or increase the number of bins. It is a multiplier to the Freedman-Draconis rule. Default: ``k_{bin}=1``
* `low_memory::Bool`: If `true` the computation needs less memory but more computation.
"""
function distance_matrix_histogram(sol::myMCSol, pars::AbstractArray{T}, distance_func, weights::AbstractArray{S}, histogram_distance::Function; matrix_distance_func::Union{Function, Nothing}=nothing, ecdf::Bool=true, k_bin::Number=1, low_memory::Bool=false) where {T,S}

    mat_elements = zeros((sol.N_mc, sol.N_mc))
    if ndims(pars) == 2
        (__, N_pars) = size(pars)
    elseif ndims(pars) == 1
        N_pars = 1
    else
        error("parameter array has more than two dimensions.")
    end

    if length(weights)!=(sol.N_meas+N_pars)
        error("Amount of weights not the same as Measures + Parameters")
    end

    if (sol.N_meas_matrix!=0) & (matrix_distance_func==nothing)
        error("There is a matrix measure in the solution but no distance func for it.")
    end

    # setup histogram edges for all histograms first, to be better comparible, they should be the same across all runs/trials
    hist_edges = []
    bin_widths = []
    for i_meas=1:sol.N_meas_dim
        data_i = get_measure(sol, i_meas)
        flat_array = collect(Iterators.flatten(data_i))

        # we use half the freedman-draconis rule because we are calculationg the IQR and max/min from _all_ values
        #bin_width = (4 * k_bin *iqr(flat_array))/(sol.N_mc^(1/3.))
        bin_width = (0.5 * k_bin *iqr(flat_array))/(sol.N_dim^(1/3.))
        minval = minimum(flat_array)
        maxval = maximum(flat_array)
        hist_edge = (minval-bin_width):bin_width:(maxval+bin_width)
        if length(hist_edge) > 100
            @warn "Very large number of Hist Bins calculated via IQR, there might be something fishy here, e.g. IQR=0. For now the Number of Bins is set to 25"
            hist_edge = range(minval - 0.1*minval, maxval + 0.1*maxval, length=25)
        end
        push!(hist_edges, hist_edge)
        push!(bin_widths, bin_width)
    end
    # do the measure loop first. this is more code, but less memory consuming as we will pre calculate the histograms

    if low_memory
        println("Calculating distance with low-memory option")
        for i_meas=1:sol.N_meas_dim # per dimension measures
            for i=1:sol.N_mc
                for j=i+1:sol.N_mc
                    mat_elements[i,j] += weights[i_meas]*histogram_distance(_compute_ecdf(sol.sol[i][i_meas], hist_edges[i_meas]),_compute_ecdf(sol.sol[j][i_meas], hist_edges[i_meas]), bin_widths[i_meas])
                end
            end
        end
    else
        for i_meas=1:sol.N_meas_dim # per dimension measures
            hist_weights = _compute_hist_weights(i_meas, sol, hist_edges, ecdf)
            for i=1:sol.N_mc
                for j=i+1:sol.N_mc
                    mat_elements[i,j] += weights[i_meas]*histogram_distance(hist_weights[i,:], hist_weights[j,:], bin_widths[i_meas])
                end
            end
        end
    end
    for i=1:sol.N_mc
        for j=1:sol.N_mc
            for i_meas = sol.N_meas_dim+1:sol.N_meas_dim+sol.N_meas_matrix
                if weights[i_meas]!=0
                    mat_elements[i,j] += matrix_distance_func(weights[i_meas]*sol.sol[i][i_meas], weights[i_meas]*sol.sol[j][i_meas])
                end
            end

            for i_meas = sol.N_meas_dim+sol.N_meas_matrix+1:sol.N_meas# global measures
                if weights[i_meas]!=0
                    mat_elements[i,j] += distance_func(weights[i_meas]*sol.sol[i][i_meas], weights[i_meas]*sol.sol[j][i_meas])
                end
            end

            for i_par=1:N_pars # parameters
                mat_elements[i,j] += distance_func(weights[sol.N_meas+i_par]*pars[i,i_par], weights[sol.N_meas+i_par]*pars[j,i_par])
            end
        end
    end
    mat_elements += transpose(mat_elements)
end
distance_matrix_histogram(sol::myMCSol, prob::myMCProblem, distance_func::Function, weights::AbstractArray, histogram_distance::Function; kwargs...) = distance_matrix_histogram(sol, parameter(prob), distance_func, weights, histogram_distance; kwargs...)

distance_matrix_histogram(sol::myMCSol, par::AbstractArray, distance_func::Function, weights::AbstractArray; kwargs...) = distance_matrix_histogram(sol, par, distance_func, weights, wasserstein_histogram_distance; kwargs...)

distance_matrix_histogram(sol::myMCSol, prob::myMCProblem, distance_func::Function, weights::AbstractArray; kwargs...) = distance_matrix_histogram(sol, parameter(prob), distance_func, weights, wasserstein_histogram_distance; kwargs...)

"""
    _compute_hist_weights(i_meas::Int, sol::myMCSol, hist_edges::AbstractArray, ecdf::Bool)

Helper function, computes histogram weights from measures for measure `i_meas`.
"""
function _compute_hist_weights(i_meas::Int, sol::myMCSol, hist_edges::AbstractArray, ecdf::Bool)
    weights = zeros(Float32, (sol.N_mc, length(hist_edges[i_meas])-1))
    for i=1:sol.N_mc
        weights[i,:] = fit(Histogram, sol.sol[i][i_meas], hist_edges[i_meas], closed=:left).weights
    end
    if ecdf
        for i=1:sol.N_mc
            weights[i,:] = ecdf_hist(weights[i,:])
        end
    end
    weights
end

function _compute_ecdf(data::AbstractArray{T}, hist_edges::AbstractArray) where T<:Number
    N_bins = length(hist_edges) - 1
    weights = zeros(eltype(data), N_bins)
    sorted_data = sort(data)
    N = length(sorted_data)
    i_bin = 1
    for i_data=1:N
        if sorted_data[i_data] > hist_edges[i_bin+1]
            i_bin += 1
            weights[i_bin] = weights[i_bin - 1]
        end
        weights[i_bin] += 1
    end
    if i_bin!=N_bins
        for i=i_bin+1:N_bins
            weights[i] = weights[i - 1]
        end
    end
    weights = weights./weights[end]
end

"""
    cluster_distance(sol::myMCSol, cluster_results::ClusteringResult, cluster_1::Int, cluster_2::Int; measures::Union{AbstractArray, Nothing}=nothing)

Does calculate the distance between the members of two cluster seperatly for each measure

# Inputs

* `sol`: Solution object
* `cluster_results`: results from the clustering
* `cluster_1`: Index of the first cluster to be analysed (noise/outlier cluster = 1)
* `cluster_2`: Index of the second cluster to be analysed
* `measures`: Which measures should be analysed, default: all.

# Output

* Array with
* Summary dictionary, mean and std of the distances
"""
function cluster_distance(sol::myMCSol, cluster_results::ClusteringResult, cluster_1::Int, cluster_2::Int; measures::Union{AbstractArray, Nothing}=nothing, distance_func=nothing, histogram_distance=nothing, matrix_distance_func=nothing)

    if measures==nothing
        measures = 1:sol.N_meas
    end

    if histogram_distance==nothing
        per_dim_distance_func = distance_func
    else
        per_dim_distance_func = histogram_distance
    end

    ca = cluster_results.assignments

    cluster_ind_1 = (ca .== (cluster_1 - 1))
    cluster_ind_2 = (ca .== (cluster_2 - 1))
    N_cluster_1 = Base.sum(cluster_ind_1)
    N_cluster_2 = Base.sum(cluster_ind_2)

    cluster_ind_1 = findall(cluster_ind_1)
    cluster_ind_2 = findall(cluster_ind_2)

    distance_funcs = [[per_dim_distance_func for i=1:sol.N_meas_dim]; [matrix_distance_func for i=1:sol.N_meas_matrix];[distance_func for i=1:sol.N_meas_global]]

    res = []
    sum = []

    # measures
    for i=1:sol.N_meas
        D = zeros(eltype(sol.sol[1][i]), N_cluster_1, N_cluster_2)
        for (ji, j) in enumerate(cluster_ind_1)
            for (ki, k) in enumerate(cluster_ind_2)
                D[ji,ki] = distance_funcs[i](sol.sol[j][i], sol.sol[k][i])
            end
        end
        push!(sum, Dict("mean"=>mean(D), "std"=>std(D)))
        push!(res, D)
    end

    return (res, sum)
end


"""

One possible histogram distance for `distance_matrix_histogram` (also the default one). It calculates the 1-Wasserstein / Earth Movers Distance between the two ECDFs by first computing the ECDF and then computing the discrete integral

``\\int_{-\\inf}^{+inf}|ECDF(hist\\_1) - ECDF(hist\\_2)| dx = \\sum_i | ECDF(hist\\_1)_i - ECDF(hist\\_2)_i | \\cdot bin\\_width``.

Returns a single (real) number. The input is the ecdf.

Adopted from [`https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html`](@ref)
"""
function wasserstein_histogram_distance(hist_1::AbstractArray{T}, hist_2::AbstractArray{T}, delta=1) where {T}
    # calculate ecdf from hist
    #ecdf_1 = ecdf_hist(hist_1)
    #ecdf_2 = ecdf_hist(hist_2)
    # as the binning is same for both histograms, we can caluclate the distance simply by the discrete integral over the difference like this:
    return sum(abs.(hist_1 .- hist_2))*delta
end

"""
    ecdf_hist(hist::Histogram)

Returns the ECDF of a histogram (normalized) as an Array.
"""
function ecdf_hist(hist::AbstractArray)
    ecdfcum = cumsum(hist)
    return ecdfcum ./ ecdfcum[end]
end


"""
    cluster_means(sol::myMCSol, clusters::ClusteringResult)

Returns the mean of each measure for each cluster.
"""
function cluster_means(sol::myMCSol, clusters::ClusteringResult)
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
    cluster_measure_mean(sol::myMCSol, clusters:ClusteringResult, i::Int)

Return the Mean of measure `i` for each cluster.
"""
function cluster_measure_mean(sol::myMCSol, clusters::ClusteringResult, i::Int)
    N_cluster = length(clusters.seeds)+1 # plus 1 -> plus "noise cluster" / not clustered points
    measure_mean = zeros(N_cluster)
    measure = get_measure(sol,i)
    for i_cluster=1:N_cluster
        measure_mean[i_cluster] = mean(measure[clusters.assignments.==(i_cluster-1),:])
    end
    measure_mean
end

"""
    cluster_measure_std(sol::myMCSol, clusters:ClusteringResult, i::Int)

Return the std of measure `i` for each cluster.
"""
function cluster_measure_std(sol::myMCSol, clusters::ClusteringResult, i::Int)
    N_cluster = length(clusters.seeds)+1 # plus 1 -> plus "noise cluster" / not clustered points
    measure_sd = zeros(N_cluster)
    measure = get_measure(sol,i)
    for i_cluster=1:N_cluster
        measure_sd[i_cluster] = std(measure[clusters.assignments.==(i_cluster-1),:])
    end
    measure_sd
end

"""
    get_measure(sol::myMCSol, i::Int, clusters::ClusteringResult, i_cluster::Int, prob::Union{MCBBProblem,Nothing}=nothing, min_par::Number=-Inf, max_par::Number=+Inf)

Get measure `i` for the `i_cluster`-th cluster with parameter values between `min_par` and `max_par`. If no `prob` is given, it ignores the parameter values
"""
function get_measure(sol::myMCSol, i::Int, clusters::ClusteringResult, i_cluster::Int, prob::Union{MCBBProblem,Nothing}=nothing, min_par::Number=-Inf, max_par::Number=+Inf)

    ca = clusters.assignments
    measure = get_measure(sol, i)
    par = parameter(prob)
    N = sol.N_mc

    cluster_ind = (ca .== (i_cluster-1))

    if prob==nothing
        par_ind = ones(Bool, N)
    else
        par_ind = (par .> min_par) .& (par .< max_par)
    end

    ind = cluster_ind .& par_ind

    if ndims(measure)==1
        return measure[ind]
    elseif ndims(measure)==2
        return measure[ind,:]
    elseif ndims(measure)==3
        return measure[ind,:,:]
    else
        error("The Measure Results has more than 3 dimensions, there is something wrong here.")
    end
end


"""
     cluster_measures(prob::myMCProblem, sol::myMCSol, clusters::ClusteringResult, window_size::AbstractArray, window_offset::AbstractArray)
     cluster_measures(prob::myMCProblem, sol::myMCSol, clusters::ClusteringResult, window_size::Number, window_offset::Number)

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
cluster_measures(prob::myMCProblem, sol::myMCSol, clusters::ClusteringResult, window_size::Number, window_offset::Number) = cluster_measures(prob, sol, clusters, [window_size], [window_offset])
function cluster_measures(prob::myMCProblem, sol::myMCSol, clusters::ClusteringResult, window_size::AbstractArray, window_offset::AbstractArray)

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
    cluster_measures_sliding_histograms(prob::myMCProblem, sol::myMCSol, clusters::ClusteringResult, i_meas::Int, window_size::Number, window_offset::Number; k_bin::Number=1)

Calculates for each window in the sliding window array a histogram of all results of meausure `i_meas` of all runs seperatly for each cluster.

Input:
* `prob::myMCProblem`: problem
* `sol::myMCSol`: solution object
* `clusters::ClusteringResult`: cluster results
* `i_meas::Int`: index of the measure to be analyzed
* `window_size::AbstractArray`: size of the window, number or Array with length according to the number of parameters
* `window_offset::AbstractArray`: size of the window, number or Array with length according to the number of parameters
* `k_bin::Number`: Bin Count Modifier. `k_bin`-times the Freedman Draconis rule is used for binning the data. Default: 1

Returns tuple with:
* `hist_vals`: N_cluster, N_windows..., N_bins - sized array with the value of the histograms for each window
* `window_mins`: left corner of the sliding windows, "x-axis-labels" of the plot
* `hist_bins`: center of the bins, "y-axis-label" of the plot

# Example Plot

    using PyPlot
    (vals, wins, edges) = cluster_measures_sliding_histograms(prob, sol, cluster_res, 1, 0.05, 0.05)
    PCP = pcolor(wins[1], hist_bins, collect(vals[1,:,:]'))
    cb = colorbar(PCP)

"""
function cluster_measures_sliding_histograms(prob::myMCProblem, sol::myMCSol, clusters::ClusteringResult, i_meas::Int, window_size::AbstractArray, window_offset::AbstractArray; k_bin::Number=1)

    N_cluster = length(clusters.seeds) + 1  # plus 1 -> plus "noise cluster" / not clustered points
    ca = clusters.assignments
    N = length(ca)
    N_par = length(ParameterVar(prob))

    N_windows, windows_mins = _sliding_window_parameter(prob, window_size, window_offset)

    measure = get_measure(sol, i_meas)

    # histograms common binning with freedman-draconis
    flat_array = collect(Iterators.flatten(measure))
    # we use double the freedman-draconis rule because we are calculationg the IQR
    # and max/min from _all_ values
    bin_width = (2. *k_bin *iqr(flat_array))/(sol.N_mc^(1/3.))
    minval = minimum(flat_array)
    maxval = maximum(flat_array)
    hist_edges = (minval-bin_width/2.):bin_width:(maxval+bin_width/2.)
    N_bins = length(hist_edges) - 1

    # these are the values
    hist_vals = zeros(N_cluster, N_windows..., N_bins)

    # cluster masks
    cluster_masks = zeros(Bool, (N_cluster, N))
    for i_cluster=0:(N_cluster - 1)
        cluster_masks[i_cluster+1,:] .= (ca .== i_cluster)
    end


    for (ip,ic) in zip(Iterators.product(windows_mins...), CartesianIndices(zeros(Int,N_windows...)))

        par_ind = ones(Bool, prob.N_mc)
        for i_par in 1:N_par
            par_ind = par_ind .& ((parameter(prob,i_par) .> ip[i_par]) .& (parameter(prob,i_par) .< (ip[i_par] + window_size[i_par])))
        end

        window_ca = ca[par_ind]
        par_ind_numbers = findall(par_ind)
        # fit histogram for each window slice

        for i_cluster=1:N_cluster
            cluster_ind = (par_ind .& cluster_masks[i_cluster,:])
            cluster_data = collect(Iterators.flatten(measure[cluster_ind, :]))
            if length(cluster_data)==0
                hist_vals[i_cluster, ic,  :] .= NaN
            else
                hist_i = normalize(fit(Histogram, cluster_data, hist_edges, closed=:left))
                hist_vals[i_cluster, ic, :] = hist_i.weights
            end
        end
    end
    (hist_vals, windows_mins, collect(hist_edges))
end
cluster_measures_sliding_histograms(prob::myMCProblem, sol::myMCSol, clusters::ClusteringResult, i_meas::Int, window_size::Number, window_offset::Number; k_bin::Number=1) = cluster_measures_sliding_histograms(prob, sol, clusters, i_meas, [window_size], [window_offset], k_bin=k_bin)

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

    ClusterICSpaces(prob::myMCProblem, sol::myMCSol, clusters::ClusteringResult; min_par::Number=-Inf, max_par::Number=Inf, nbins::Int64=20)

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

    function ClusterICSpaces(prob::myMCProblem, sol::myMCSol, clusters::ClusteringResult; min_par::Number=-Inf, max_par::Number=Inf, nbins::Int64=20)

        if length(ParameterVar(prob))>1
            error("Not yet implemented for systems with more than one parameter")
        end

        N_cluster = length(clusters.seeds)+1 # plus 1 -> plus "noise cluster" / not clustered points
        N_dim = sol.N_dim

        icp = prob.ic
        pars = parameter(prob)
        ca = clusters.assignments

        # collect the data for each cluster and dimension
        cross_dim_means = [[] for i=1:N_cluster]
        cross_dim_stds = [[] for i=1:N_cluster]
        cross_dim_skews = [[] for i=1:N_cluster]

        data = [[[] for i=1:N_dim+1] for i=1:N_cluster] # +1 for the parameter
        for i=1:sol.N_mc
            if (pars[i] > min_par) & (pars[i] < max_par)
                i_cluster = ca[i] + 1  # plus 1 -> plus "noise cluster" / not clustered points
                for i_dim=1:N_dim # ICs
                    push!(data[i_cluster][i_dim],icp[i,i_dim])
                end

                i_mean, i_std = mean_and_std(icp[i,:])
                i_skew = skewness(icp[i,:], i_mean)
                push!(cross_dim_means[i_cluster], i_mean)
                push!(cross_dim_stds[i_cluster], i_std)
                push!(cross_dim_skews[i_cluster], i_skew)
                push!(data[i_cluster][N_dim+1],pars[i,1]) # parameter
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
    cluster_n_noise(clusters::ClusteringResult)

Returns the number of points assignt to the "noise" cluster (somehow this is not automaticlly returned by the routine of Clustering.jl).
"""
function cluster_n_noise(clusters::ClusteringResult)
    count = 0
    for i=1:length(clusters.assignments)
        if clusters.assignments[i]==0
            count += 1
        end
    end
    count
end

"""
    cluster_membership(par::AbstractArray, clusters::ClusteringResult)

Calculates the proportion of members for each cluster for all parameter values.
"""
function cluster_membership(par::AbstractArray, clusters::ClusteringResult)
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
# -  par needs to be 1d mapping each run to its parameter value, e.g. par[:,end]
# this method uses a sliding window over the parameter axis.
# should be used when parameters are randomly generated.
# - normalize: normalize by number of parameters per window
function cluster_membership(par::AbstractArray, clusters::ClusteringResult, window_size::Number, window_offset::Number, normalize::Bool=true)
    error("This used to be an old version of cluster_membership, please use cluster_membership(prob::myMCProb, clusters::ClusteringResult, window_size, window_offset) now")
end

"""
    cluster_membership(prob::myMCProblem, clusters::ClusteringResult, window_size::AbstractArray, window_offset::AbstractArray, normalize::Bool=true)
    cluster_membership(prob::myMCProblem, clusters::ClusteringResult, window_size::Number, window_offset::Number, normalize::Bool=true)

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
function cluster_membership(prob::myMCProblem, clusters::ClusteringResult, window_size::AbstractArray, window_offset::AbstractArray, normalize::Bool=true)

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
cluster_membership(prob::myMCProblem, clusters::ClusteringResult, window_size::Number, window_offset::Number, normalize::Bool=true) = cluster_membership(prob, clusters, [window_size], [window_offset], normalize)

"""
    measure_on_parameter_sliding_window

Does calculate measures (per cluster) on parameter sliding windows. This routine is called by `cluster_membership` and `cluster_measures` but can also be used for plotting measures on the parameter grid manually.

_ATTENTION_: If a cluster has no members within a window the value is set to `NaN`. This should simply omit these points from beeing plotted (while `missing` and `nothing` are currently not compatible with most plotting packages).

    measure_on_parameter_sliding_window(prob::myMCProblem, sol::myMCSol, i::Int, clusters::ClusteringResult, window_size::Number, window_offset::Number)

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
measure_on_parameter_sliding_window(prob::myMCProblem, sol::myMCSol, i::Int, clusters::ClusteringResult, window_size::Number, window_offset::Number) = measure_on_parameter_sliding_window(prob, sol, i, clusters, [window_size], [window_offset])
measure_on_parameter_sliding_window(prob::myMCProblem, sol::myMCSol, i::Int, window_size::AbstractArray, window_offset::AbstractArray) = measure_on_parameter_sliding_window(prob, sol, i, zeros(Int,prob.N_mc), 1, window_size, window_offset)
measure_on_parameter_sliding_window(prob::myMCProblem, sol::myMCSol, i::Int, clusters::ClusteringResult, window_size::AbstractArray, window_offset::AbstractArray) = measure_on_parameter_sliding_window(prob, sol, i, clusters.assignments, length(clusters.seeds) + 1, window_size, window_offset)
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
    get_trajectory(prob::MCBBProblem, sol::MCBBSol, clusters::ClusteringResult, i::Int; only_sol::Bool=true)

Solves and returns a trajectory that is classified in cluster `i`. Randomly selects one IC/Parameter configuration, so that mulitple executions of this routine will yield different results! If `only_sol==true` it returns only the solution, otherwise it returns a tuple `(solution, problem, i_run)` where `i_run` is the number of the trial in `prob` and `sol`.

# Example

Plot with e.g

    using PyPlot
    IM = imshow(Matrix(get_trajectory(prob,sol,db_res,1)), aspect=2)
    ylabel("System Dimension i")
    xlabel("Time t")
    cb = colorbar(IM, orientation="horizontal")
    cb[:ax][:set_xlabel]("Colorbar: Im(z)", rotation=0)
"""
function get_trajectory(prob::MCBBProblem, sol::MCBBSol, clusters::ClusteringResult, i::Int; only_sol::Bool=true)
    i_sol = rand(findall(clusters.assignments .== (i-1)))
    prob_i = prob.p.prob_func(prob.p.prob, i_sol, false)
    sol_i = sol.solve_command(prob_i)
    if only_sol
        return sol_i
    else
        return (sol_i, prob_i, i_sol)
    end
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
