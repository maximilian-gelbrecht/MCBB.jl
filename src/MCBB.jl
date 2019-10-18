module MCBB

"""
    myMCProblem

Abstract type for all problem types defined in this library. Note that the DifferentialEquations.jl problem types are _not_  supertypes of this types.
"""
abstract type myMCProblem end

"""
    myMCSol

Abstract type for all problem types defined in this library. Note that the DifferentialEquations.jl problem types are _not_  supertypes of this types.
"""
abstract type myMCSol end

# export all functions declared
export kuramoto_parameters, kuramoto, kuramoto_network_parameters, kuramoto_network, logistic_parameters, logistic, henon_parameters, henon, roessler_parameters, roessler_network, lotka_volterra, lotka_volterra_parameters, second_order_kuramoto_chain, second_order_kuramoto_chain_parameters, remake_second_order_kuramoto_chain_paramaters
export second_order_kuramoto, second_order_kuramoto_parameters
export non_local_kuramoto_ring_parameters, non_local_kuramoto_ring
export myMCProblem, DEMCBBProblem, myMCSol, sort, sort!, parameter, show_results, solve
export ParameterVar
export tsave_array
export ClusterICSpaces
export ContinuationProblem
export ContinuationSolution, DEMCBBSol
export setup_ic_par_mc_problem, define_new_problem, eval_ode_run, check_inf_nan
export distance_matrix, distance_matrix_dense, weighted_norm
export order_parameter

# internal functions, also exported for testing
export empirical_1D_KL_divergence_pc, empirical_1D_KL_divergence_hist, wasserstein_hist
export _compute_wasserstein_hist
export wasserstein_ecdf
#export curve_entropy
export cluster_measure_mean, cluster_measure_std
export k_dist, cluster_measures, cluster_means, cluster_n_noise, cluster_membership
export custom_problem_new_parameters
export normalize, get_measure
export KNN_dist, KNN_dist_relative
export compute_parameters
export DEParameters
export solve_euler_inf, tsave_array
export measure_on_parameter_sliding_window
export ParameterVar, ParameterVarArray, ParameterVarFunc, OneDimParameterVar, MultiDimParameterVar, MultiDimParameterVarFunc, MultiDimParameterVarArray
export MCBBProblem, MCBBSol
export CustomProblem, CustomSolution, CustomMonteCarloProblem, CustomMCBBSolution, CustomMCBBProblem
export get_trajectory
export cluster_measures_sliding_histograms
export distance_matrix_histogram, wasserstein_histogram_distance, ecdf_hist
export stuart_landau_sathiyadevi_pars, stuart_landau_sathiyadevi!
export BBClusterResult, bbcluster
export correlation_hist, correlation, correlation_ecdf
export cluster_distance

export AbstractDistanceMatrix, DistanceMatrix, DistanceMatrixHist
export ClusterMembershipResult, ClusterMeasureResult, ClusterMeasureHistogramResult
export distance_matrix_sparse, distance_matrix_mmap
export NonzeroSparseMatrix
# Contains example systems
include("systems.jl")

# all functions and methods needed to help you setup EnsembleProblems over the combined initial conditions - parameter space
include("setup_mc_prob.jl")
include("eval_mc_prob.jl")

# all function and needed needed to evaluate (DBSCAN)-clustering
include("eval_clustering.jl")
include("bif_analysis.jl")
include("custom_mc_prob.jl")
include("bbclustering.jl")
include("plots.jl")
include("sparse_clustering.jl")
end
