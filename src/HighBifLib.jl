module HighBifLib

abstract type myMCProblem end
abstract type myMCSol end

# export all functions declared
export kuramoto_parameters, kuramoto, kuramoto_network_parameters, kuramoto_network, logistic_parameters, logistic, henon_parameters, henon, roessler_parameters, roessler_network, lotka_volterra, lotka_volterra_parameters, second_order_kuramoto_chain, second_order_kuramoto_chain_parameters, remake_second_order_kuramoto_chain_paramaters
export myMCProblem, BifAnaMCProblem, myMCSol, sort, sort!, parameter, show_results, solve
export ParameterVar
export tsave_array
export BifAnalysisProblem
export BifAnalysisSolution, BifMCSol
export setup_ic_par_mc_problem, define_new_problem, eval_ode_run, check_inf_nan
export distance_matrix, distance_matrix_dense, weighted_norm
export order_parameter

# internal functions, also exported for testing
export empirical_1D_KL_divergence_pc, empirical_1D_KL_divergence_hist, wasserstein_hist
export _compute_wasserstein_hist
export wasserstein_ecdf
export curve_entropy
export k_dist, cluster_measures, cluster_n_noise, cluster_membership


# Contains example systems
include("systems.jl")

# all functions and methods needed to help you setup MonteCarloProblems over the combined initial conditions - parameter space
include("setup_mc_prob.jl")

# all function needed to evaluate the solutions of the MonteCarloProblem
include("eval_mc_prob.jl")

# all function and needed needed to evaluate (DBSCAN)-clustering
include("eval_clustering.jl")

include("bif_analysis.jl")

end
