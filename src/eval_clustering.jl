###########
##### Results Evaluation functions
###########
using Distributions, Clustering, StatsBase, SparseArrays
import Mmap
import Clustering.dbscan
#using PairwiseListMatrices

"""
    abstract type AbstractDistanceMatrix{T} <: AbstractArray{T,2} end

Abstract Datatype for all Distance Matrix types. Currently, there are within MCBB:
    * [`DistanceMatrix`](@ref)
    * [`DistanceMatrixHist`](@ref)
"""
abstract type AbstractDistanceMatrix{T} <: AbstractArray{T,2} end

"""
    DistanceMatrix{T}

Type for distance matrices. This type should behave just like any `AbstractArray{T,2}`. There's a [`convert`](@ref) to `AbstractArray{T,2}`.

It also holds additional information about the distance calculation.

# Fields (and constructor)

* `data::AbstractArray{T,2}`: The actual distance matrix
* `weights::AbstractArray{T,1}`: The weights that were used to compute it
* `distance_func::Function`: The function that was used to compute it
* `relative_parameter::Bool`: Was the parameter rescaled?
"""
mutable struct DistanceMatrix{T,S} <: AbstractDistanceMatrix{T}
    data::AbstractArray{T,2}
    weights::AbstractArray{S,1}
    distance_func::Function
    matrix_distance_func::Union{Function, Nothing}
    relative_parameter::Bool
end
_distance_func(D::DistanceMatrix) = D.distance_func

"""
    DistanceMatrixHist{T}

Type for distance matrices which were computed using Histograms. This type should behave just like any `AbstractArray{T,2}`. There's a [`convert`](@ref) to `AbstractArray{T,2}`.

It also holds additional information about the distance calculation.

# Fields (and constructor)

* `data::AbstractArray{T,2}`: The actual distance matrix
* `weights::AbstractArray{T,1}`: The weights that were used to compute it
* `distance_func::Function`: The function that was used to compute the distance between the global measures
* `matrix_distance_func::Union{Function, Nothing}`: The function that was used to compute it
* `relative_parameter::Bool`: Was the parameter rescaled?
* `histogram_distance::Function`: Function used to compute the histogram distance
* `hist_edges`: Array of arrays/ranges with all histogram edges
* `bin_width`: Array of all histogram bin widths
* `ecdf`: Was the ECDF used in the distance computation?
* `k_bin`: Additional factor in bin_width computation

"""
mutable struct DistanceMatrixHist{T,S} <: AbstractDistanceMatrix{T}
    data::AbstractArray{T,2}
    weights::AbstractArray{S,1}
    distance_func
    matrix_distance_func
    relative_parameter::Bool
    histogram_distance
    hist_edges
    bin_width
    ecdf::Bool
    k_bin::Number
end

Base.size(dm::AbstractDistanceMatrix) = size(dm.data)
Base.getindex(dm::AbstractDistanceMatrix, i::Int) = getindex(dm.data, i)
Base.getindex(dm::AbstractDistanceMatrix, I...) = getindex(dm.data, I...)
Base.setindex!(dm::AbstractDistanceMatrix, v, i::Int) = setindex!(dm.data, v, i)
Base.setindex!(dm::AbstractDistanceMatrix, v, I::Vararg) = setindex!(dm.data, v, I)

Base.convert(::AbstractArray{T,2}, dm::AbstractDistanceMatrix{T}) where T<:Number = dm.data
_distance_func(D::DistanceMatrixHist) = D.histogram_distance

Clustering.dbscan(dm::AbstractDistanceMatrix{T}, eps::Number, k::Int) where T<:Number = dbscan(dm.data, eps, k)

"""
     distance_matrix(sol::myMCSol, prob::myMCProblem, distance_func::Function, weights::AbstractArray; matrix_distance_func::Union{Function, Nothing}=nothing, histogram_distance_func::Union{Function, Nothing}=wasserstein_histogram_distance, relative_parameter::Bool=false, histograms::Bool=false, use_ecdf::Bool=true, k_bin::Number=1, bin_edges::AbstractArray)

Calculate the distance matrix between all individual solutions.

# Histogram Method

If it is called with the `histograms` flag `true`, computes for each run in the solution `sol` for each measure a histogram of the measures of all system dimensions. The binning of the histograms is computed with Freedman-Draconis rule and the same across all runs for each measure.

The distance matrix is then computed given a suitable histogram distance function `histogram_distance` between these histograms.

This is intended to be used in order to avoid symmetric configurations in larger systems to be distinguished from each other. Example: Given a system with 10 identical oscillators. Given this distance calculation a state where oscillator 1-5 are synchronized and 6-10 are not syncronized would be in the same cluster as a state where oscillator 6-10 are synchronized and 1-5 are not synchronized. If you don't want this kind of behaviour, use the regular `distance_matrix` function.

# Arguments

* `sol`: solution
* `prob`: problem
* `distance_func`: The actual calculating the distance between the measures/parameters of each solution with each other. Signature should be `(measure_1::Union{Array,Number}, measure_2::Union{Array,Number}) -> distance::Number. Example and default is `(x,y)->sum(abs.(x .- y))`.
* `weights`: Instead of the actual measure `weights[i_measure]*measure` is handed over to `distance_func`. Thus `weights` need to be ``N_{meas}+N_{par}`` long array.

## Kwargs

* `relative_parameter`: If true, the paramater values during distance calcuation is rescaled to [0,1]
* `histograms::Bool`: If true, the distance calculation is based on [`distance_matrix_histogram`](@ref) with the default histogram distance [`wasserstein_histogram_distance`](@ref).
* `histogram_distance_func`: The distance function between two histograms. Default is [`wasserstein_histogram_distance`](@ref).
* `matrix_distance_func`: The distance function between two matrices or arrays or length different from ``N_{dim}``. Used e.g. for Crosscorrelation.
* `ecdf::Bool` if true the `histogram_distance` function gets the empirical cdfs instead of the histogram
* `k_bin::Int`: Multiplier to increase (``k_{bin}>1``) or decrease the bin width and thus decrease or increase the number of bins. It is a multiplier to the Freedman-Draconis rule. Default: ``k_{bin}=1``
* `nbin_default::Int`: If the IQR is very small and thus the number of bins larger than `nbin_default`, the number of bins is set back to `nbin_default` and the edges and width adjusted accordingly.
* `nbin::Int` If specified, ingore all other histogram binning calculation and use nbin bins for the histograms.
* `bin_edges::AbstractArray`: If specified ignore all other histogram binning calculations and use this as the edges of the histogram (has to have one more element than bins, hence all edges). Needs to be an Array with as many elements as measures, if one wants automatic binning for one observables, this element of the array has to be `nothing`. E.g.: `[1:1:10, nothing, 2:0.5:5]`.

Returns an instance of [`DistanceMatrix`](@ref) or [`DistanceMatrixHist`](@ref)
"""
function distance_matrix(sol::myMCSol, prob::myMCProblem, distance_func::Function, weights::AbstractArray; matrix_distance_func::Union{Function, Nothing}=nothing, histogram_distance_func::Union{Function, Nothing}=wasserstein_histogram_distance, relative_parameter::Bool=false, histograms::Bool=false, use_ecdf::Bool=true, k_bin::Number=1, nbin_default::Int=50, nbin::Union{Int, Nothing}=nothing, bin_edges::Union{AbstractArray, Nothing}=nothing)

    N_pars = length(ParameterVar(prob))

    pars = parameter(prob)
    par_c = copy(pars)

    if nbin != nothing
        @warn "nbin amount specified, all usual histogram binning function will not be used"
    end

    if bin_edges == nothing
        bin_edges = [nothing for i=1:sol.N_meas_dim]
    end

    if (sol.N_meas_matrix!=0) & (matrix_distance_func==nothing)
        error("There is a matrix measure in the solution but no distance func for it.")
    end

    if relative_parameter
        par_c = _relative_parameter(prob)
    end

    if histograms

        # setup histogram edges for all histograms first, to be better comparible, they should be the same across all runs/trials
        hist_edges = []
        bin_widths = []
        for i_meas=1:sol.N_meas_dim
            hist_edge, bin_width = _compute_hist_edges(i_meas, sol, k_bin, nbin_default=nbin_default, nbin=nbin, bin_edges=bin_edges[i_meas])
            push!(hist_edges, hist_edge)
            push!(bin_widths, bin_width)
        end

        if histogram_distance_func == nothing
            error("Histogram method chosen, but no histogram_distance_func given.")
        end
    end

    mat_elements = zeros((sol.N_mc, sol.N_mc))

    for i_meas=1:sol.N_meas_dim
        if histograms
            _compute_distance!(mat_elements, sol, i_meas, histogram_distance_func, hist_edges[i_meas], bin_widths[i_meas], weights[i_meas], use_ecdf)
        else
            _compute_distance!(mat_elements, sol, i_meas, distance_func, weights[i_meas])
        end
    end
    for i_meas =sol.N_meas_dim+1:sol.N_meas_dim+sol.N_meas_matrix
        _compute_distance!(mat_elements, sol, i_meas, matrix_distance_func, weights[i_meas])
    end
    for i_meas = sol.N_meas_dim+sol.N_meas_matrix+1:sol.N_meas
        _compute_distance!(mat_elements, sol, i_meas, distance_func, weights[i_meas])
    end
    for i_par=1:N_pars
        for i=1:sol.N_mc
            for j=i+1:sol.N_mc
                mat_elements[i,j] += weights[sol.N_meas+i_par]*distance_func(par_c[i,i_par], par_c[j,i_par])
            end
        end
    end

    mat_elements += transpose(mat_elements)

    if sum(isnan.(mat_elements))>0
        @warn "There are some elements NaN in the distance matrix"
    end
    if sum(isinf.(mat_elements))>0
        @warn "There are some elements Inf in the distance matrix"
    end

    if histograms
        return DistanceMatrixHist(mat_elements, weights, distance_func, matrix_distance_func, relative_parameter, histogram_distance_func, hist_edges, bin_widths, use_ecdf, k_bin)
    else
        return DistanceMatrix(mat_elements, weights, distance_func, matrix_distance_func, relative_parameter)
    end
end
distance_matrix(sol::myMCSol, prob::myMCProblem, weights::AbstractArray; kwargs...) = distance_matrix(sol, prob, (x,y)->sum(abs.(x .- y)), weights; kwargs...)

"""
    _compute_distance!(D::AbstractArray, sol::myMCSol, i_meas::Int, distance_func::Function, weight::Number=1)

Computes (inplace) the distance matrix contribution from measure `i_meas`.
"""
function _compute_distance!(D::AbstractArray{T,2}, sol::myMCSol, i_meas::Int, distance_func::Function, weight::Number=1) where T <: Number
    for i=1:sol.N_mc
        for j=i+1:sol.N_mc
            D[i,j] += weight * distance_func(sol.sol[i][i_meas], sol.sol[j][i_meas])
        end
    end
end

"""
    _compute_distance!(D::AbstractArray, sol::myMCSol, i_meas::Int, distance_func::Function, hist_edges::AbstractArray, bin_width::Number, weight::Number=1, use_ecdf::Bool=true)

Computes (inplace) the distance matrix contribution from measure `i_meas` with the histogram method.
"""
function _compute_distance!(D::AbstractArray{T,2}, sol::myMCSol, i_meas::Int, distance_func::Function, hist_edges::AbstractArray, bin_width::Number, weight::Number=1, use_ecdf::Bool=true) where T <: Number
    hist_weights = _compute_hist_weights(i_meas, sol, hist_edges, use_ecdf)
    for i=1:sol.N_mc
        for j=i+1:sol.N_mc
            D[i,j] += weight * distance_func(hist_weights[i,:], hist_weights[j,:], bin_width)
        end
    end
end

"""
    _compute_distance!(D::AbstractArray{T,1}, i::Int, sol::myMCSol, i_meas::Int, distance_func::Function, weight::Number=1) where T <: Number

Computes (inplace) the `i`-th row of the distance matrix contribution from measure `i_meas`.

"""
function _compute_distance!(D::AbstractArray{T,1}, i::Int, sol::myMCSol, i_meas::Int, distance_func::Function, weight::Number=1) where T <: Number
    for j=1:sol.N_mc
        D[j] += weight * distance_func(sol.sol[i][i_meas], sol.sol[j][i_meas])
    end
end

"""
    _compute_distance!(D::AbstractArray{T,2}, sol::myMCSol, i_meas::Int, distance_func::Function, hist_edges::AbstractArray, hist_weights::AbstractArray, bin_width::Number, weight::Number=1) where T <: Number

Computes (inplace) the `i`-th row of the distance matrix contribution from measure `i_meas` with the histogram method.
"""
function _compute_distance!(D::AbstractArray{T,1}, i::Int, sol::myMCSol, i_meas::Int, distance_func::Function, hist_edges::AbstractArray, hist_weights::AbstractArray, bin_width::Number, weight::Number=1) where T <: Number
    for j=1:sol.N_mc
        D[j] += weight * distance_func(hist_weights[i,:], hist_weights[j,:], bin_width)
    end
end

"""
    compute_distance(sol::myMCSol, i_meas::Int, distance_func::Function; use_histograms::Bool=false, use_ecdf::Bool=true, k_bin::Number=1, bin_edges::AbstractRange)

Computes a (part of the) distance matrix for only a single measure `i_meas`. Follows otherwise the same logic as [`distance_matrix`](@ref) but returns the matrix as an `Array{T,2}`.
"""
function compute_distance(sol::myMCSol, i_meas::Int, distance_func::Function; use_histograms::Bool=false, use_ecdf::Bool=true, k_bin::Number=1, nbin::Union{Int, Nothing}=nothing, bin_edges::Union{AbstractArray, Nothing}=nothing)
    D = zeros((sol.N_mc, sol.N_mc))
    if use_histograms
        hist_edge, bin_width = _compute_hist_edges(i_meas, sol, k_bin, nbin=nbin)
        _compute_distance!(D, sol, i_meas, distance_func, hist_edge, bin_width, use_ecdf)
    else
        _compute_distance!(D, sol, i_meas, distance_func)
    end
    return D + transpose(D)
end

"""
    distance_matrix_mmap(sol::myMCSol, prob::myMCProblem, distance_func::Function, weights::AbstractArray; matrix_distance_func::Union{Function, Nothing}=nothing, histogram_distance_func::Union{Function, Nothing}=wasserstein_histogram_distance, relative_parameter::Bool=false, histograms::Bool=false, use_ecdf::Bool=true, k_bin::Number=1, nbin_default::Int=50, el_type=Float32, save_name="mmap-distance-matrix.bin")

Computes the distance matrix like [`distance_matrix`](@ref) but uses memory-maped arrays. Use this if the distance matrix is too large for the memory of your computer. Same inputs as [`distance_matrix`](@ref), but with added `el_type` that determines the eltype of the saved matrix and `save_name` the name of the file on the hard disk.

Due to the restriction of memory-maped arrays saving and loading distance matrices computed like this with JLD2 will only work within a single machine. A way to reload these matrices / transfer them, is [`reload_mmap_distance_matrix`](@ref).

"""
function distance_matrix_mmap(sol::myMCSol, prob::myMCProblem, distance_func::Function, weights::AbstractArray; matrix_distance_func::Union{Function, Nothing}=nothing, histogram_distance_func::Union{Function, Nothing}=wasserstein_histogram_distance, relative_parameter::Bool=false, histograms::Bool=false, use_ecdf::Bool=true, k_bin::Number=1, nbin_default::Int=50, nbin::Union{Int, Nothing}=nothing, bin_edges::Union{AbstractArray, Nothing}=nothing, el_type=Float32, save_name="mmap-distance-matrix.bin")

    N_pars = length(ParameterVar(prob))

    pars = parameter(prob)
    par_c = copy(pars)

    if (sol.N_meas_matrix!=0) & (matrix_distance_func==nothing)
        error("There is a matrix measure in the solution but no distance func for it.")
    end

    if bin_edges == nothing
        bin_edges = [nothing for i=1:sol.N_meas_dim]
    end

    if relative_parameter
        par_c = _relative_parameter(prob)
    end

    if histograms
        # setup histogram edges for all histograms first, to be better comparible, they should be the same across all runs/trials
        hist_edges = []
        bin_widths = []
        for i_meas=1:sol.N_meas_dim
            hist_edge, bin_width = _compute_hist_edges(i_meas, sol, k_bin, nbin_default=nbin_default, nbin=nbin, bin_edges=bin_edges[i_meas])
            push!(hist_edges, hist_edge)
            push!(bin_widths, bin_width)
        end

        if histogram_distance_func == nothing
            error("Histogram method chosen, but no histogram_distance_func given.")
        end
    end

    # open mmap file
    f = open(save_name, "w+")
    write(f, sol.N_mc)
    write(f, sol.N_mc)

    #mat_elements = zeros((sol.N_mc, sol.N_mc))
    for i=1:sol.N_mc
        row_elements = zeros(el_type, sol.N_mc)

        for i_meas = 1:sol.N_meas_dim
            if histograms
                hist_weights = _compute_hist_weights(i_meas, sol, hist_edges[i_meas], use_ecdf)
                _compute_distance!(row_elements, i, sol, i_meas, histogram_distance_func, hist_edges[i_meas], hist_weights, bin_widths[i_meas], weights[i_meas])
            else
                _compute_distance!(row_elements, i, sol, i_meas, distance_func, weights[i_meas])
            end
        end


        for i_meas = sol.N_meas_dim+1:sol.N_meas_dim+sol.N_meas_matrix
            _compute_distance!(row_elements, i, sol, i_meas, matrix_distance_func, weights[i_meas])
        end

        for i_meas = sol.N_meas_dim+sol.N_meas_matrix+1:sol.N_meas
            _compute_distance!(row_elements, i, sol, i_meas, distance_func, weights[i_meas])
        end

        for i_par=1:N_pars
            for j=1:sol.N_mc
                row_elements[j] += weights[sol.N_meas+i_par]*distance_func(par_c[i,i_par], par_c[j,i_par])
            end
        end

        if sum(isnan.(row_elements))>0
            @warn "There are some elements NaN in the distance matrix"
        end
        if sum(isinf.(row_elements))>0
            @warn "There are some elements Inf in the distance matrix"
        end

        # save the row
        write(f, row_elements)
    end
    close(f)

    # do the mmap work
    f = open(save_name)
    m = read(f, Int)
    n = read(f, Int)
    mat_elements = Mmap.mmap(f, Matrix{el_type}, (m,n))

    if histograms
        return DistanceMatrixHist(mat_elements, weights, distance_func, matrix_distance_func, relative_parameter, histogram_distance_func, hist_edges, bin_widths, use_ecdf, k_bin)
    else
        return DistanceMatrix(mat_elements, weights, distance_func, matrix_distance_func, relative_parameter)
    end
end
distance_matrix_mmap(sol::myMCSol, prob::myMCProblem, weights::AbstractArray; kwargs...) = distance_matrix_mmap(sol, prob, (x,y)->sum(abs.(x .- y)), weights; kwargs...)

"""
    reload_mmap_distance_matrix(old_instance::AbstractDistanceMatrix, binary_file; el_type=Float32)

Reloads a corrupted/old instance of `AbstractDistanceMatrix` that is using `Mmap` from the `Mmap` binary file saved at `binary_file` with eltype
"""
function reload_mmap_distance_matrix(old_instance::DistanceMatrix, binary_file; el_type=Float32)
    f = open(binary_file)
    m = read(f, Int)
    n = read(f, Int )
    mat_elements = Mmap.mmap(f, Matrix{el_type}, (m,n))

    return DistanceMatrix(mat_elements, old_instance.weights, old_instance.distance_func, old_instance.matrix_distance_func, old_instance.relative_parameter)
end

function reload_mmap_distance_matrix(old_instance::DistanceMatrixHist, binary_file; el_type=Float32)
    f = open(binary_file)
    m = read(f, Int)
    n = read(f, Int )
    mat_elements = Mmap.mmap(f, Matrix{el_type}, (m,n))

    return DistanceMatrixHist(mat_elements, old_instance.weights, old_instance.distance_func, old_instance.matrix_distance_func, old_instance.relative_parameter, old_instance.histogram_distance, old_instance.hist_edges, old_instance.bin_width, old_instance.ecdf, old_instance.k_bin)
end

"""
    metadata!(dm::AbstractDistanceMatrix, )

Sets the input [`AbstractDistanceMatrix`] matrix itself empty, thus only containing metadata. This is usefull if the matrix itself is already saved otherwise (like with Mmap).
"""
function metadata!(dm::AbstractDistanceMatrix)
    dm.data = Array{eltype(dm.data)}(undef, 0, 0)
end

"""
    distance_matrix_sparse(sol::myMCSol, prob::myMCProblem, distance_func::Function, weights::AbstractArray; matrix_distance_func::Union{Function, Nothing}=nothing, histogram_distance_func::Union{Function, Nothing}=wasserstein_histogram_distance, relative_parameter::Bool=false, histograms::Bool=false, use_ecdf::Bool=true, k_bin::Number=1, nbin_default::Int=50, nbin::Union{Int, Nothing}=nothing, bin_edges::Union{AbstractArray, Nothing}=nothing, sparse_threshold::Number=Inf, el_type=Float32, check_inf_nan::Bool=true)

Computes the distance matrix sparse. Same arguments as [`distance_matrix`](@ref) with extra arguments

    * `sparse_threshold`: Only distances smaller than this value are saved
    * `check_inf_nan`: Only performs the Inf/NaN check if true.

"""
function distance_matrix_sparse(sol::myMCSol, prob::myMCProblem, distance_func::Function, weights::AbstractArray; matrix_distance_func::Union{Function, Nothing}=nothing, histogram_distance_func::Union{Function, Nothing}=wasserstein_histogram_distance, relative_parameter::Bool=false, histograms::Bool=false, use_ecdf::Bool=true, k_bin::Number=1, nbin_default::Int=50, nbin::Union{Int, Nothing}=nothing, bin_edges::Union{AbstractArray, Nothing}=nothing, sparse_threshold::Number=Inf, el_type=Float32, check_inf_nan::Bool=true)

    N_pars = length(ParameterVar(prob))

    pars = parameter(prob)
    par_c = copy(pars)

    if nbin != nothing
        @warn "nbin amount specified, all usual histogram binning function will not be used"
    end

    if bin_edges == nothing
        bin_edges = [nothing for i=1:sol.N_meas_dim]
    end

    if (sol.N_meas_matrix!=0) & (matrix_distance_func==nothing)
        error("There is a matrix measure in the solution but no distance func for it.")
    end

    if relative_parameter
        par_c = _relative_parameter(prob)
    end

    if histograms

        # setup histogram edges for all histograms first, to be better comparible, they should be the same across all runs/trials
        hist_edges = []
        bin_widths = []
        for i_meas=1:sol.N_meas_dim
            hist_edge, bin_width = _compute_hist_edges(i_meas, sol, k_bin, nbin_default=nbin_default, nbin=nbin, bin_edges=bin_edges[i_meas])
            push!(hist_edges, hist_edge)
            push!(bin_widths, bin_width)
        end

        if histogram_distance_func == nothing
            error("Histogram method chosen, but no histogram_distance_func given.")
        end
    end

    mat_elements = spzeros(el_type, sol.N_mc, sol.N_mc)
    sparse_threshold = el_type(sparse_threshold)
    if histograms
        function dfunc(i,j,i_meas)
            hweights = zeros(Float32, (2, length(hist_edges[i_meas])-1))
            println(hist_edges[i_meas])
            hweights[1,:] = fit(Histogram, sol.sol[i][i_meas], hist_edges[i_meas], closed=:left).weights
            hweights[2,:] = fit(Histogram, sol.sol[j][i_meas], hist_edges[i_meas], closed=:left).weights
            if use_ecdf
                for ii in 1:2
                    hweights[ii,:] = ecdf_hist(hweights[ii,:])
                end
            end
            weights[i_meas] * histogram_distance_func(hweights[1,:], hweights[2,:], bin_widths[i_meas])
        end
    else
        dfunc(i,j,i_meas) = weights[i_meas] * distance_func(sol.sol[i][i_meas], sol.sol[j][i_meas])
    end

    dfuncs = []
    for i_meas=1:sol.N_meas_dim
        push!(dfuncs, (i,j) -> dfunc(i,j,i_meas))
    end
    for i_meas=sol.N_meas_dim+1:sol.N_meas_dim+sol.N_meas_matrix
        push!(dfuncs, (i,j) -> weights[i_meas]*matrix_distance_func(sol.sol[i][i_meas], sol.sol[j][i_meas]))
    end
    for i_meas=sol.N_meas_dim+sol.N_meas_matrix+1:sol.N_meas
        push!(dfuncs, (i,j) -> weights[i_meas]*distance_func(sol.sol[i][i_meas], sol.sol[j][i_meas]))
    end

    for i=1:sol.N_mc
        for j=i+1:sol.N_mc
            d_val = el_type(0.)
            d_val_sparse = true
            if (i==1) & (j==3)
                println(d_val)
            end
            for i_meas=1:sol.N_meas
                d_val += dfuncs[i_meas](i,j)

                if (i==1) & (j==3)
                    println(d_val)
                end
                if d_val > sparse_threshold
                    d_val_sparse = false
                    break
                end
            end

            if d_val_sparse
                mat_elements[i,j] = d_val - sparse_threshold
            end
        end
    end
    println("///")
    println(mat_elements[1,3])
    mat_elements += (transpose(mat_elements) + spdiagm(0=>-1*sparse_threshold*ones(sol.N_mc)))
    println("///")
    println(mat_elements[1,3])
    if check_inf_nan
        if sum(isnan.(mat_elements))>0
            @warn "There are some elements NaN in the distance matrix"
        end
        if sum(isinf.(mat_elements))>0
            @warn "There are some elements Inf in the distance matrix"
        end
    end

    mat_elements = NonzeroSparseMatrix(mat_elements, sparse_threshold)
    println("///")
    println(mat_elements[1,3])
    if histograms
        return DistanceMatrixHist(mat_elements, weights, distance_func, matrix_distance_func, relative_parameter, histogram_distance_func, hist_edges, bin_widths, use_ecdf, k_bin)
    else
        return DistanceMatrix(mat_elements, weights, distance_func, matrix_distance_func, relative_parameter)
    end
end
distance_matrix_sparse(sol::myMCSol, prob::myMCProblem, weights::AbstractArray; kwargs...) = distance_matrix_sparse(sol, prob, (x,y)->sum(abs.(x .- y)), weights; kwargs...)


#(sol::myMCSol, prob::myMCProblem, distance_func::Function, weights::AbstractArray; matrix_distance_func::Union{Function, Nothing}=nothing, histogram_distance_func::Union{Function, Nothing}=nothing, relative_parameter::Bool=false, histograms::Bool=false, use_ecdf::Bool=true, k_bin::Number=1)


"""
    _relative_parameter(prob::myMCProblem)

Helper function that rescales the parameter to [0,1]. Returns an Array with the rescaled parameter.
"""
function _relative_parameter(prob::myMCProblem)
    N_pars = length(ParameterVar(prob))
    par = parameter(prob)
    par_c = copy(par)
    for i_par=1:N_pars
        min_par = minimum(par[:,i_par])
        max_par = maximum(par[:,i_par])
        #par_range = max_par - min_par
        par_rel = (par[:,i_par] .- min_par)./max_par
        par_c[:,i_par] = par_rel
    end
    return par_c
end

"""
    _compute_hist_weights(i_meas::Int, sol::myMCSol, hist_edges::AbstractArray, ecdf::Bool)

Helper function, computes histogram weights from measures for measure `i_meas`.
"""
function _compute_hist_weights(i_meas::Int, sol::myMCSol, hist_edge::AbstractArray, use_ecdf::Bool)
    weights = zeros(Float32, (sol.N_mc, length(hist_edge)-1))
    for i=1:sol.N_mc
        weights[i,:] = fit(Histogram, sol.sol[i][i_meas], hist_edge, closed=:left).weights
    end
    if use_ecdf
        for i=1:sol.N_mc
            weights[i,:] = ecdf_hist(weights[i,:])
        end
    end
    weights
end

"""
    _compute_hist_edges(i_meas::Int, sol::myMCSol, k_bin::Number, nbin_default::Int=50)

Helper function, computes the edges of the histograms. Uses Freedman-Draconis rule. `k_bin` is an additional prefactor to the computed bin width.

If the IQR is very small and thus the number of bins larger than `nbin_default`, the number of bins is set back to `nbin_default` and the edges and width adjusted accordingly.
"""
_compute_hist_edges(i_meas::Int, sol::myMCSol, k_bin::Number; kwargs...) = _compute_hist_edges(get_measure(sol, i_meas), i_meas, sol.N_dim, k_bin; kwargs...)
function _compute_hist_edges(data::AbstractArray, i_meas::Int, N_dim::Int, k_bin::Number; nbin_default::Int=50, nbin::Union{Int, Nothing}=nothing, bin_edges::Union{AbstractRange, Nothing}=nothing)
    if (bin_edges!=nothing) & (nbin!=nothing)
        error("bin_edges and nbin in kwargs. Please choose only one.")
    end

    minval = minimum(data)
    maxval = maximum(data)

    if (nbin==nothing) & (bin_edges==nothing)
        bin_width = freedman_draconis_bin_width(data, N_dim, k_bin)

        if bin_width==0
            @warn string("Bin Width at measure number ",i_meas,", calculated via IQR is 0. For now the Number of Bins is set to ", nbin_default)
            hist_edge = range(minval - 0.1*minval, maxval + 0.1*maxval, length=nbin_default)
            bin_width = hist_edge[2] - hist_edge[1]
        else

            hist_edge = (minval-bin_width):bin_width:(maxval+bin_width)
            if length(hist_edge) > nbin_default
                @warn string("Very large number of Hist Bins at measure number ",i_meas,", calculated via IQR, there might be something fishy here, e.g. IQR=0. For now the Number of Bins is set to ", nbin_default)
                hist_edge = range(minval - 0.1*minval, maxval + 0.1*maxval, length=nbin_default)
                bin_width = hist_edge[2] - hist_edge[1]
            end
        end
    elseif nbin!=nothing
        @assert 1 < nbin

        bin_width = (maxval - minval)/(nbin-1)
        hist_edge = (minval-bin_width):bin_width:(maxval+bin_width)
    elseif bin_edges!=nothing
        hist_edge = bin_edges
        bin_width = bin_edges[2] - bin_edges[1]
    end
    return hist_edge, bin_width
end

freedman_draconis_bin_width(data::AbstractArray{T,1}, N_dim::Int, k_bin::Number) where T <: Number = 0.5 * k_bin *iqr(data)/N_dim^(1/3.)
freedman_draconis_bin_width(data::AbstractArray{T,2}, N_dim::Int, k_bin::Number) where T <: Number = freedman_draconis_bin_width(collect(Iterators.flatten(data)), N_dim, k_bin)

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
    cluster_distance(sol::myMCSol, D::AbstractDistanceMatrix, cluster_results::ClusteringResult,  cluster_1::Int, cluster_2::Int; measures::Union{AbstractArray, Nothing}=nothing, distance_func=nothing, histogram_distance=nothing, matrix_distance_func=nothing, k_bin::Number=1)

Does calculate the distance between the members of two cluster seperatly for each measure

# Inputs

* `sol`: Solution object
* `D`: distance matrix from [`distance_matrix`](@ref)
* `cluster_results`: results from the clustering
* `cluster_1`: Index of the first cluster to be analysed (noise/outlier cluster = 1)
* `cluster_2`: Index of the second cluster to be analysed
* `measures`: Which measures should be analysed, default: all.

# Output

* Array with
* Summary dictionary, mean and std of the distances
"""
function cluster_distance(sol::myMCSol, dm::AbstractDistanceMatrix, cluster_results::ClusteringResult, cluster_1::Int, cluster_2::Int; measures::Union{AbstractArray, Nothing}=nothing)

    if measures==nothing
        measures = 1:sol.N_meas
    end

    ca = cluster_results.assignments

    cluster_ind_1 = (ca .== (cluster_1 - 1))
    cluster_ind_2 = (ca .== (cluster_2 - 1))
    N_cluster_1 = Base.sum(cluster_ind_1)
    N_cluster_2 = Base.sum(cluster_ind_2)

    cluster_ind_1 = findall(cluster_ind_1)
    cluster_ind_2 = findall(cluster_ind_2)

    distance_funcs = [[_distance_func(dm) for i=1:sol.N_meas_dim]; [dm.matrix_distance_func for i=1:sol.N_meas_matrix];[dm.distance_func for i=1:sol.N_meas_global]]
    hist_flag = (typeof(dm) <: DistanceMatrixHist)

    res = []
    sum = []

    # measures
    for i in measures
        D = zeros(eltype(sol.sol[1][i]), N_cluster_1, N_cluster_2)

        if hist_flag & (i <= sol.N_meas_dim) # histogram distance
            hist_weights = _compute_hist_weights(i, sol, dm.hist_edges[i], dm.ecdf)

            for (ji, j) in enumerate(cluster_ind_1)
                for (ki, k) in enumerate(cluster_ind_2)
                    D[ji,ki] = distance_funcs[i](hist_weights[j,:], hist_weights[k,:], dm.bin_width[i])
                end
            end
        else  # regular distance
            for (ji, j) in enumerate(cluster_ind_1)
                for (ki, k) in enumerate(cluster_ind_2)
                    D[ji,ki] = distance_funcs[i](sol.sol[j][i], sol.sol[k][i])
                end
            end
        end
        push!(sum, Dict("mean"=>mean(D), "std"=>std(D)))
        push!(res, D)
    end

    return (res, sum)
end


"""

One possible histogram distance for `distance_matrix_histogram` (also the default one). It calculates the 1-Wasserstein / Earth Movers Distance between the two ECDFs by first computing the ECDF and then computing the discrete integral

``\\int_{-\\infty}^{+\\infty}|ECDF(hist\\_1) - ECDF(hist\\_2)| dx = \\sum_i | ECDF(hist\\_1)_i - ECDF(hist\\_2)_i | \\cdot bin\\_width``.

Returns a single (real) number. The input is the ecdf.

Adopted from [`https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html`](@ref)
"""
function wasserstein_histogram_distance(hist_1::AbstractArray{T}, hist_2::AbstractArray{T}, delta::Number=1) where {T}
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
    if length(clusters.counts) == 0
        @warn "No clusters found"
        return false
    end

    N_cluster = length(clusters.seeds)+1 # plus 1 -> plus "noise cluster" / not clustered points
    N_dim = sol.N_dim
    mean_measures = zeros((N_cluster,sol.N_meas_dim,N_dim))

    cluster_counts = [cluster_n_noise(clusters); clusters.counts]
    for i_sol=1:sol.N_mc
        for i_meas=1:sol.N_meas_dim
            mean_measures[clusters.assignments[i_sol]+1,i_meas,:] += sol.sol[i_sol][i_meas] / cluster_counts[clusters.assignments[i_sol]+1]
        end
    end
    mean_measures
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
    get_measure(sol::myMCSol, i::Int, clusters::ClusteringResult, i_cluster::Int, prob::Union{MCBBProblem,Nothing}=nothing, min_par::Number=-Inf, max_par::Number=+Inf, i_par::Int=1)

Get measure `i` for the `i_cluster`-th cluster with parameter values between `min_par` and `max_par`. If no `prob` is given, it ignores the parameter values. In case a multidimensional setup is used, uses the `i_par`-th Parameter.
"""
function get_measure(sol::myMCSol, i::Int, clusters::ClusteringResult, i_cluster::Int, prob::Union{MCBBProblem,Nothing}=nothing, min_par::Number=-Inf, max_par::Number=+Inf, i_par::Int=1)

    ca = clusters.assignments
    measure = get_measure(sol, i)

    N = sol.N_mc

    cluster_ind = (ca .== (i_cluster-1))

    if prob==nothing
        par_ind = ones(Bool, N)
    else

        par = parameter(prob)

        if length(ParameterVar(prob)) > 1
            par = par[:,i_par]
        end

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

Returns an instance of [`ClusterMeasureResult`](@ref) with fields:
* `par`: the center value of the sliding windows, in case multiple parameter are being varied, it is a meshgrid.
* `cluster_measures`: (per dimension) measures on the parameter grid
* `cluster_measures_global`: global measures on the parameter grid

# Plot:

* The `i`-th measure can be plotted with `plot(res::ClusterMeasureResult, i::Int, kwargs...)`

* A single cluster and measure can be plotted with `plot(res::ClusterMeasureResult, i_meas::Int, i_cluster::Int, kwargs...)`.
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
    _multidim = (N_par > 1) ? true : false
    ClusterMeasureResult(p_windows, cluster_measures, cluster_measures_global, _multidim)
end

"""
    ClusterMeasureResult

Results of [`cluster_measures`](@ref).

# Fields:
* `par`
* `cluster_measures`
* `cluster_measures_global`

# Plot:

The `i`-th measure of the `j-`th cluster can be plotted with `plot(res::ClusterMeasureResult, i::Int, j::Int, kwargs...)`

"""
struct ClusterMeasureResult
    par
    cluster_measures
    cluster_measures_global
    multidim_flag
end

"""
    cluster_measures_sliding_histograms(prob::myMCProblem, sol::myMCSol, clusters::ClusteringResult, i_meas::Int, window_size::Number, window_offset::Number; kwargs...)

Calculates for each window in the sliding window array a histogram of all results of meausure `i_meas` of all runs seperatly for each cluster.

Input:
* `prob::myMCProblem`: problem
* `sol::myMCSol`: solution object
* `clusters::ClusteringResult`: cluster results
* `i_meas::Int`: index of the measure to be analyzed
* `window_size::AbstractArray`: size of the window, number or Array with length according to the number of parameters
* `window_offset::AbstractArray`: size of the window, number or Array with length according to the number of parameters

Keyword arguments
* `k_bin::Number`: Bin Count Modifier. `k_bin`-times the Freedman Draconis rule is used for binning the data. Default: 1
* `normalization_mode::Symbol`, normalization mode applied to Histograms. Directly handed over to [`normalize`](@ref).
* `nbin::Int`: Uses nbins for the histograms instead of the (automatic) Freedman Draconis rule
* `bin_edges::AbstractRange`: Uses these edges for the histograms.
* `state_filter::AbstractArray`: Only use these system dimension as the basis for the computation, default: all. Attention: if the evalation function already used a state_filter this will be refering only to the system dimension that were measured.

Returns an instance of [`ClusterMeasureHistogramResult`](@ref) with fields:
* `hist_vals`: N_cluster, N_windows..., N_bins - sized array with the value of the histograms for each window
* `par`: midpoint of the sliding windows, "x-axis-labels" of the plot
* `hist_bins`: center of the bins, "y-axis-label" of the plot

Can be plotted with `plot(res::ClusterMeasureHistogramResult, kwargs...)`. See [`ClusterMeasureHistogramResult`](@ref) for details.

"""
function cluster_measures_sliding_histograms(prob::myMCProblem, sol::myMCSol, clusters::ClusteringResult, i_meas::Int, window_size::AbstractArray, window_offset::AbstractArray; state_filter::Union{AbstractArray, Nothing}=nothing, k_bin::Number=1, normalization_mode::Symbol=:probability, nbin::Union{Int, Nothing}=nothing, bin_edges::Union{AbstractRange, Nothing}=nothing)

    N_cluster = length(clusters.seeds) + 1  # plus 1 -> plus "noise cluster" / not clustered points
    ca = clusters.assignments
    N = length(ca)
    N_par = length(ParameterVar(prob))

    N_windows, windows_mins = _sliding_window_parameter(prob, window_size, window_offset)

    measure = get_measure(sol, i_meas, state_filter=state_filter)

    # histograms common binning with freedman-draconis
    hist_edges, bin_width = _compute_hist_edges(measure, i_meas, sol.N_dim, k_bin, nbin=nbin, bin_edges=bin_edges)
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
                hist_i = normalize(fit(Histogram, cluster_data, hist_edges, closed=:left), mode=normalization_mode)
                hist_vals[i_cluster, ic, :] = hist_i.weights
            end
        end
    end
    _multidim = (N_par > 1) ? true : false
    ClusterMeasureHistogramResult(hist_vals, windows_mins, collect(hist_edges[1:end-1]) .+ (bin_width/2.), _multidim)
end
cluster_measures_sliding_histograms(prob::myMCProblem, sol::myMCSol, clusters::ClusteringResult, i_meas::Int, window_size::Number, window_offset::Number; kwargs...) = cluster_measures_sliding_histograms(prob, sol, clusters, i_meas, [window_size], [window_offset]; kwargs...)

"""
    ClusterMeasureHistogramResult

Stores results of [`cluster_measures_sliding_histograms`](@ref).

# Fields

* `hist_vals`: N_cluster, N_windows..., N_bins - sized array with the value of the histograms for each window
* `par`: midpoint of the sliding windows, "x-axis-labels" of the plot
* `hist_edges`: center of the bins, "y-axis-label" of the plot
* `multidim_flag`

# Plot

Can be plotted with `plot(res::ClusterMeasureHistogramResult, i, kwargs...)`. With `i` being the number of the cluster.
"""
struct ClusterMeasureHistogramResult
    hist_vals
    par
    hist_edges
    multidim_flag
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

Returns an instance [`ClusterMembershipResult`](@ref) with fields:
* `par`: the center value of the sliding windows, in case multiple parameter are being varied, it is a meshgrid.
* `data`: members of the clusters on the parameter grid
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
    ClusterMembershipResult(par, memberships, false)
end

"""
    cluster_membership(prob::myMCProblem, clusters::ClusteringResult, window_size::AbstractArray, window_offset::AbstractArray; normalize::Bool=true, min_members::Int=0)
    cluster_membership(prob::myMCProblem, clusters::ClusteringResult, window_size::Number, window_offset::Number; normalize::Bool=true,  min_members::Int=0)

Calculates the proportion of members for each cluster within a parameter sliding window.

* `prob`: problem
* `sol`: solution of `prob`
* `clusters`: results from a DBSCAN run.
* `window_size`: Size of the window. In case multiple paramaters being varied has to be an array.
* `window_offset`: Offset of the sliding window. In case multiple paramaters being varied has to be an array.

Returns an instance [`ClusterMembershipResult`](@ref) with fields:
* `par`: the center value of the sliding windows, in case multiple parameter are being varied, it is a meshgrid.
* `data`: members of the clusters on the parameter grid

The results can be plotted with directly with `plot(results, kwargs...)`. See [`ClusterMembershipResult`](@ref) for details on the plotting and operationg on this type.
"""
function cluster_membership(prob::myMCProblem, clusters::ClusteringResult, window_size::AbstractArray, window_offset::AbstractArray; normalize::Bool=true, min_members::Int=0)

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

    if min_members > 0
        minpts_list = []
        if cluster_n_noise(clusters) > min_members
            push!(minpts_list, 1)
        end

        for i=1:(N_cluster - 1)
            if clusters.counts[i] > min_members
                push!(minpts_list, i+1)
            end
        end
        memberships = getindex(memberships,[Colon() for i=1:(ndims(memberships)-1)]...,minpts_list)
    end

    # return
    if N_par == 1
        flag = false
    else
        flag = true
    end

    ClusterMembershipResult(parameter_mesh, memberships, flag)
end
cluster_membership(prob::myMCProblem, clusters::ClusteringResult, window_size::Number, window_offset::Number; kwargs...) = cluster_membership(prob, clusters, [window_size], [window_offset]; kwargs...)

"""
    ClusterMembershipResult{T,S}

Stores the results of [`cluster_membership`](@ref) and can be used for [`ClusterMembershipPlot`](@ref).

# Fields
* `par`: Parameter Array or Mesh
* `data`: Cluster Membership data on `par`-Parameter grid.
* `multidim_flag`: Is the experiment multidimensional?

# Plot
    plot(cm::ClusterMembershipResult, kwargs...)

Does plot the [`ClusterMembershipResult`](@ref). Uses Plot recipes and thus hands over all kwargs possible from Plots.jl.

## Hints

The order of the labels for the legend is reversed.

## Additional Kwargs

* `plot_index`: Range or Array with the indices of the clusters to be plotted. Default: all.

# Additonal operation defined
    * can be indexed
    * can be sorted, [`Base.sort!(cm::ClusterMembershipResult; ignore_first::Bool)`](@ref)
    * can be summed, [`Base.sum(cm::ClusterMembershipResult, indices::AbstractArray{Int,1})`](@ref)
"""
mutable struct ClusterMembershipResult
    par::AbstractArray
    data::AbstractArray
    multidim_flag::Bool
end

Base.size(cm::ClusterMembershipResult) = size(cm.data)
Base.getindex(cm::ClusterMembershipResult, i::Int) = ClusterMembershipResult(cm.par, cm.data[:,i:i], cm.multidim_flag)
Base.getindex(cm::ClusterMembershipResult, I::AbstractArray) = ClusterMembershipResult(cm.par, cm.data[:,I], cm.multidim_flag)
Base.setindex!(cm::ClusterMembershipResult, v, i::Int) = setindex!(cm.data, v, i)
Base.setindex!(cm::ClusterMembershipResult, v, I::Vararg) = setindex!(cm.data, v, I)

"""
    sort!(cm::ClusterMembershipResult; ignore_first::Bool=false)

Sorts `cm` inplace by the count of members of the clusters from low to high. If `ignore_first` is true, the first cluster (with DBSCAN this is the outlier cluster) is ignored while sorting and remains the first cluster.
"""
function Base.sort!(cm::ClusterMembershipResult; ignore_first::Bool=false)
    tmp = deepcopy(cm)
    N_cluster = size(cm.data)[end]
    Nc = zeros(N_cluster)
    for i=1:N_cluster
        Nc[i] = sum(getindex(cm, [Colon() for i=1:(ndims(cm)-1)]..., i))
    end
    if ignore_first
        sortind = [1; (sortperm(Nc[2:end]).+1)]
    else
        sortind = sortperm(Nc)
    end

    cm.data = getindex(tmp, [Colon() for i=1:(ndims(cm)-1)]..., sortind)
end

"""
    Base.sum(cm::ClusterMembershipResult, indices::AbstractArray{Int,1})

Returns a `ClusterMembershipResult` with all `indices` clusters summed together.
"""
function Base.sum(cm::ClusterMembershipResult, indices::AbstractArray{Int,1})
    if cm.multidim_flag
        return ClusterMembershipResult(cm.par, reshape(sum([cm.data[:,:,i] for i in indices]),:,1), cm.multidim_flag)
    else
        return ClusterMembershipResult(cm.par, reshape(sum([cm.data[:,i] for i in indices]),:,1), cm.multidim_flag)
    end
end


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

    get_trajectory(prob::MCBBProblem, sol::MCBBSol, i::Int, only_sol::Bool=true)

Solves problem `i` and returns a trajectory. If `only_sol==true` it returns only the solution, otherwise it returns a tuple `(solution, problem, i_run)` where `i_run` is the number of the trial in `prob` and `sol`.

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
    return get_trajectory(prob, sol, i_sol, only_sol=only_sol)
end

function get_trajectory(prob::MCBBProblem, sol::MCBBSol, i::Int; only_sol::Bool=true)
    prob_i = prob.p.prob_func(prob.p.prob, i, false)
    sol_i = sol.solve_command(prob_i)
    if only_sol
        return sol_i
    else
        return (sol_i, prob_i, i)
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
