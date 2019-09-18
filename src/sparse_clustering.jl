# modification of clustering.jl's DBSCAN to work with sparse matrices

using Clustering
import Clustering.dbscan
import Clustering._dbscan
import Clustering._dbs_region_query
import Clustering._dbs_expand_cluster!


# missing: sparse routine for distgance matrix
using SparseArrays

abstract type AbstractNonzeroSparseMatrix{T,S} <: AbstractSparseMatrix{T,S} end

"""
    NonzeroSparseMatrix{T,S,V}

Wraps around a regular SparseMatrix to set a default value different from zero.

So far, this is a very minimal implementation that only supports the very basic operations. This may be extended in the future.

# Initialization

    NonzeroSparseMatrix(data::AbstractArray{T,2}, indices::BitArray{2}, default_value::Number) where T<:Number

* `data`: Input data, mmap or regular dense array
* `indices`: of the elements that are saved / are not the default value
* `default_value`: default value that replaces the zero from regular sparse matrices.

NonzeroSparseMatrix(data::AbstractArray{T,2}, condition, default_value::Number) where T<:Number

* `data`: Input data, mmap or regular dense array
* `condition`: Function `(data[i,j]->true/false)` that sets which elements are saved. Saves memory as it does not use a full BitArray index matrix.
* `default_value`: default value that replaces the zero from regular sparse matrices.

"""
struct NonzeroSparseMatrix{Tv,Ti<:Integer,S} <: AbstractNonzeroSparseMatrix{Tv,Ti}
    spmat::AbstractSparseMatrix{Tv,Ti}
    value::S
end

function NonzeroSparseMatrix(data::AbstractArray{T,2}, indices::BitArray{2}, default_value::Number) where T<:Number
    spmat = spzeros(T, size(data)...)
    spmat[indices] .= data[indices] .- T(default_value)
    NonzeroSparseMatrix(spmat, T(default_value))
end

function NonzeroSparseMatrix(data::AbstractArray{T,2}, condition, default_value::Number) where T<:Number
    spmat = spzeros(T, size(data)...)
    defval = T(default_value)
    for i=1:size(data,1)
        for j=1:size(data,2)
            if condition(data[i,j])
                spmat[i,j] = data[i,j] - defval
            end
        end
    end
    NonzeroSparseMatrix(spmat, defval)
end

Base.getindex(mat::NonzeroSparseMatrix, I...) = Base.getindex(mat.spmat, I...) .+ mat.value
Base.getindex(mat::NonzeroSparseMatrix, i::Int) = Base.getindex(mat.spmat, i::Int) .+ mat.value
Base.size(mat::NonzeroSparseMatrix) = Base.size(mat.spmat)

Base.setindex!(mat::NonzeroSparseMatrix, v, i::Int) = Base.setindex!(mat.spmat, v .- mat.value, i)
Base.setindex!(mat::NonzeroSparseMatrix, v, I...) = Base.setindex!(mat.spmat, v .- mat.value, I...)

Base.similar(mat::NonzeroSparseMatrix) = NonzeroSparseMatrix(Base.similar(mat.spmat), mat.value)

## main algorithm

"""
    dbscan(D::DenseMatrix, eps::Real, minpts::Int) -> DbscanResult

Perform DBSCAN algorithm using the distance matrix `D`.

# Arguments
The following options control which points would be considered
*density reachable*:
  - `eps::Real`: the radius of a point neighborhood
  - `minpts::Int`: the minimum number of neighboring points (including itself)
     to qualify a point as a density point.
"""
function dbscan(D::AbstractNonzeroSparseMatrix, eps::Real, minpts::Int)
    # check arguments
    n = size(D, 1)
    size(D, 2) == n || throw(ArgumentError("D must be a square matrix ($(size(D)) given)."))
    n >= 2 || throw(ArgumentError("At least two data points are required ($n given)."))
    eps > 0 || throw(ArgumentError("eps must be a positive value ($eps given)."))
    minpts >= 1 || throw(ArgumentError("minpts must be positive integer ($minpts given)."))

    # invoke core algorithm
    _dbscan(D, convert(eltype(D), eps), minpts, 1:n)
end

function _dbscan(D::AbstractNonzeroSparseMatrix, eps::T, minpts::Int, visitseq::AbstractVector{Int}) where T<:Real
    n = size(D, 1)

    # prepare
    seeds = Int[]
    counts = Int[]
    assignments = zeros(Int, n)
    visited = zeros(Bool, n)
    k = 0

    # main loop
    for p in visitseq
        if assignments[p] == 0 && !visited[p]
            visited[p] = true
            nbs = _dbs_region_query(D, p, eps)
            if length(nbs) >= minpts
                k += 1
                cnt = _dbs_expand_cluster!(D, k, p, nbs, eps, minpts, assignments, visited)
                push!(seeds, p)
                push!(counts, cnt)
            end
        end
    end

    # make output
    return DbscanResult(seeds, assignments, counts)
end

## key steps

function _dbs_region_query(D::AbstractNonzeroSparseMatrix, p::Int, eps::T) where T<:Real
    n = size(D,1)
    nbs = Int[]
    dists = view(D,:,p)
    for i = 1:n
        @inbounds if dists[i] < eps
            push!(nbs, i)
        end
    end
    return nbs::Vector{Int}
end

function _dbs_expand_cluster!(D::AbstractNonzeroSparseMatrix,           # distance matrix
                              k::Int,                      # the index of current cluster
                              p::Int,                      # the index of seeding point
                              nbs::Vector{Int},            # eps-neighborhood of p
                              eps::T,                      # radius of neighborhood
                              minpts::Int,                 # minimum number of neighbors of a density point
                              assignments::Vector{Int},    # assignment vector
                              visited::Vector{Bool}) where T<:Real       # visited indicators
    assignments[p] = k
    cnt = 1
    while !isempty(nbs)
        q = popfirst!(nbs)
        if !visited[q]
            visited[q] = true
            qnbs = _dbs_region_query(D, q, eps)
            if length(qnbs) >= minpts
                for x in qnbs
                    if assignments[x] == 0
                        push!(nbs, x)
                    end
                end
            end
        end
        if assignments[q] == 0
            assignments[q] = k
            cnt += 1
        end
    end
    return cnt
end
