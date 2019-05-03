# Here, the Basin Bifurcation Clustering Algorithm is in developing
# the code is based on the DBSCAN code from Clustering.jl
import Clustering.ClusteringResult

mutable struct BBClusterResult <: ClusteringResult
    seeds::Vector{Int}          # starting points of clusters, size (k,)
    assignments::Vector{Int}    # assignments, size (n,)
    counts::Vector{Int}         # number of points in each cluster, size (k,)
end

"""
    bbcluster(D::AbstractDistanceMatrix, dplus::AbstractVector{T}, dminus::AbstractVector{T}, pars::AbstractVector{T}, p_eps::T, minpts::Int; k::S=1.5, par_distance_func::Union{Function, Nothing}=nothing) where {T,S}<:Real

Performs the BBClustering, a modified DBSCAN clustering adjusted for Basin Bifurcation Analysis

Inputs:

* `D`: Distance Matrix (`NxN`)
* `dplus`: Response of Distance Measure at `p+\\delta p`
* `dminus`: Response of Distance Measure at `p-\\delta p`
* `pars`: Parameter vector
* `delta_p`: Used to estimate response `dplus` and `dminus`
* `p_eps`: Epsilon Parameter, only points with parameters closer than `p_eps` are connected.
* `minpts`: Minimum number of points for a cluster, otherwise outlier
* `k`: Paramater for the clustering, should be `1 < k < 2`
* `par_distance_func`: Distance function for parameters, check: `par_distance_func(pars[i],pars[j]) < p_eps`

    bbcluster(D::AbstractArray, prob::MCBBProblem, sol::MCBBSol, delta_p::T; p_eps::Union{Nothing,T}=nothing, minpts::Int=1, k::Number=1.5, par_distance_func::Union{Function,Nothing}=nothing) where T<:Real

Convenience wrapper of the above defined function with ['MCBBProblem'](@ref) and ['MCBBSol'](@ref) as inputs. Default value for `p_eps` is five times the mean parameter difference.
"""
function bbcluster(D::AbstractDistanceMatrix, dplus::AbstractVector, dminus::AbstractVector, pars::AbstractVector, delta_p::Real, p_eps::Real, minpts::Int=1; k::Number=1.5, par_distance_func::Union{Function, Nothing}=nothing)
    n = size(D, 1)
    size(D, 2) == n || error("D must be a square matrix.")
    n >= 2 || error("There must be at least two points.")
    k > 0 || error("k must be a positive real value.")
    p_eps > 0 || error("p_eps must be a positive real value.")
    minpts >= 1 || error("minpts must be a positive integer.")
    size(dplus, 1) == n || error("dplus must have the same length as rows/columns in D")
    size(dminus, 1) == n || error("dminus must have the same length as rows/columns in D")
    size(pars, 1) == n || error("pars must have the same length as rows/columns in D")

    if par_distance_func==nothing
        par_distance_func = (x,y) -> abs(x - y)
    end
    # invoke core algorithm
    _bbcluster(D, dplus, dminus, pars, delta_p, p_eps, minpts, k, 1:n, par_distance_func)
end

function bbcluster(D::AbstractArray, prob::MCBBProblem, sol::MCBBSol, delta_p::T; p_eps::Union{Nothing,T}=nothing, minpts::Int=1, k::Number=1.5, par_distance_func::Union{Function,Nothing}=nothing) where T<:Real

    # lets use 5 times the mean parameter difference as a default value for p_eps
    if p_eps==nothing
        p_eps = 10. * mean(abs.(diff(parameter(prob))))
    end

    bbcluster(D, get_measure(sol, sol.N_meas), get_measure(sol, sol.N_meas - 1), parameter(prob), delta_p, p_eps, minpts, k=k, par_distance_func=par_distance_func)
end

function _bbcluster(D::AbstractArray{T}, dplus::AbstractVector{T}, dminus::AbstractVector{T}, pars::AbstractVector{T}, delta_p::T, p_eps::T, minpts::Int, k::Real, visitseq::AbstractVector{Int}, par_distance_func::Function) where T<:Real
    n = size(D, 1)

    # prepare
    seeds = Int[]
    counts = Int[]
    assignments = zeros(Int, n)
    visited = zeros(Bool, n)
    kval = k / 2.

    k_c = 0 # cluster number
    # main loop
    for p in visitseq
        if assignments[p] == 0 && !visited[p]
            #@printf "p=%d p_eps=%f kval=%f" p p_eps kval
            visited[p] = true
            nbs = _bb_region_query(D, dplus, dminus, pars, delta_p, p_eps, p, kval, par_distance_func)
            if length(nbs) >= minpts
                k_c += 1
                cnt = _bb_expand_cluster!(D, dplus, dminus, pars, delta_p, k_c, p, nbs, p_eps, kval, minpts, assignments, visited, par_distance_func)
                #@printf "k_c=%d cnt=%d minpts=%d" k_c cnt minpts
                push!(seeds, p)
                push!(counts, cnt)
            end
        end
    end

    # make output
    return BBClusterResult(seeds, assignments, counts)
end

## key steps
# dbg: seems to work
function _bb_region_query(D::AbstractArray{T}, dplus::AbstractVector{T}, dminus::AbstractVector{T}, pars::AbstractVector{T}, delta_p::T, p_eps::T, p::Int, k::T, par_distance_func::Function) where T<:Real
    n = size(D,1)
    nbs = Int[]
    dists = view(D,:,p)
    dpm_min = min(dplus[p],dminus[p])

    for i = 1:n
        dist_par = par_distance_func(pars[i], pars[p])
        if dist_par < p_eps
            rescaled_dist = dists[i]*(delta_p/dist_par)
            @inbounds if rescaled_dist <= k*(dpm_min + min(dplus[i],dminus[i]))
                push!(nbs, i)
            end
        end
    end
    return nbs::Vector{Int}
end

function _bb_expand_cluster!(D::AbstractArray{T},           # distance matrix
                              dplus::AbstractVector{T},
                              dminus::AbstractVector{T},
                              pars::AbstractVector{T},
                              delta_p::T,
                              k_c::Int,                    # the index of current cluster
                              p::Int,                      # the index of seeding point
                              nbs::Vector{Int},            # p_eps-neighborhood of p
                              p_eps::T,                      # radius of neighborhood
                              kval::T,
                              minpts::Int,                 # minimum number of neighbors of a density point
                              assignments::Vector{Int},    # assignment vector
                              visited::Vector{Bool},       # visited indicators
                              par_distance_func::Function) where T<:Real
    assignments[p] = k_c
    cnt::Int = 1
    while !isempty(nbs)
        q = popfirst!(nbs)
        if !visited[q]
            visited[q] = true
            qnbs = _bb_region_query(D, dplus, dminus, pars, delta_p, p_eps, q, kval, par_distance_func)
            if length(qnbs) >= minpts
                for x in qnbs
                    if assignments[x] == 0
                        push!(nbs, x)
                    end
                end
            end
        end
        if assignments[q] == 0
            assignments[q] = k_c
            cnt += 1
        end
    end
    return cnt
end
