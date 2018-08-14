#####
## Library
## Structered as follows:
## 1. Systems - various structs and functions that can be run
## 2. ODE Evaluation functions - various functions that help evaluate the system runs
## 3. Results Evalulation - various functions that help evaluate all the combined results
####

module HighBifLib

using DifferentialEquations, Miniball, Distances, Parameters, Clustering, Interpolations, LightGraphs
import StatsBase, Distributions
import DifferentialEquations.solve # this needs to be directly importet in order to extend it with our own solve() for our own problem struct

# export all functions declared
export kuramoto_parameters, kuramoto, kuramoto_network_parameters, kuramoto_network, logistic_parameters, logistic, henon_parameters, henon, roessler_parameters, roessler_network, lotka_volterra, lotka_volterra_parameters
export myMCProblem, EqMCProblem, myMCSol
export setup_ic_par_mc_problem, eval_ode_run, eval_ode_run_repeat, eval_ode_run_inf, check_inf_nan
export distance_matrix, weighted_norm
export order_parameter

# internal functions, also exported for testing
export empirical_1D_KL_divergence, ecdf_pc

export curve_entropy
export k_dist, cluster_measures

#################
##### Systems
#################
# The systems are all defined here, because it makes saving and loading results much easier (with e.g. JLD2) if they are part of the library as well.
# TO-DO: also defining all the jacobians would improve the performance of some of the solvers used
#################
abstract type DEParameters end

@with_kw struct logistic_parameters <: DEParameters  # with_kw enables default values and keyword initializing and more importantly a very convinient reconstruct routine!
    r::Float64
end

function logistic(u_next, u, p::logistic_parameters, t)
    u_next[1] = p.r * u[1] * ( 1. - u[1])
end

@with_kw struct henon_parameters <: DEParameters
    a::Float64
    b::Float64
end

function henon(u_next, u, p::henon_parameters, t)
    xnext[1] = 1.0 - p.a*x[1]^2 + x[2]
    xnext[2] = p.b*x[1]
end

@with_kw struct kuramoto_parameters <: DEParameters
    K::Number # coupling strength
    w::Array{Float64} # eigenfrequencies
    N::Int  # Number of Oscillators
end

# fully connected kuramoto model
function kuramoto(du, u, p::kuramoto_parameters, t)
    for i=1:p.N
        du[i] = 0.
        for j=1:p.N
            du[i] += sin.(u[j] - u[i])
        end
        du[i] *= p.K/p.N
        du[i] += p.w[i]
    end
end

@with_kw struct kuramoto_network_parameters <: DEParameters # unfortunatly one can only subtype from abstract types in Julia
    K::Number # coupling strength
    w::Array{Float64} # eigenfrequencies
    N::Int  # Number of Oscillators
    A # adjacency_matrix, either sparse or dense
end

# kuramoto model on a network
function kuramoto_network(du, u, p::kuramoto_network_parameters, t)
    for i=1:p.N
        du[i] = 0.
        for j=1:p.N
            if p.A[i,j]!=0
                du[i] += p.A[i,j]*sin.(u[j] - u[i])
            end
        end
        du[i] *= p.K/p.N
        du[i] += p.w[i]
    end
end

# order_parameter
# Kuratomo Order Parameter
function order_parameter(u::AbstractArray, N::Int)
    1./N*sqrt((sum(sin.(u)))^2+(sum(cos.(u)))^2)
end

# roessler parameters for Roessler Network
@with_kw struct roessler_parameters <: DEParameters
    a::Array{Float64}
    b::Array{Float64}
    c::Array{Float64}
    K::Float64 # coupling
    L::AbstractSparseMatrix # Laplacian
    N::Int # number of nodes
end

function roessler_parameters(a, b, c, K, k::Int64, p::Float64, N)
    G = watts_strogatz(N, k, p)
    L = laplacian_matrix(G)
    roessler_parameters(a,b,c,K,L,N)
end

function roessler_network(du, u, p::roessler_parameters, t)
    for i=1:p.N
        ii = 3*i
        du[ii-2] = - u[ii-1] - u[ii]
        for j=1:p.N
            if p.L[i,j]!=0
                du[ii-2] -= p.K*p.L[i,j]*du[ii-2]
            end
        end
        du[ii-1] = u[ii-2] + p.a[i]*u[ii-1]
        du[ii] = p.b[i] + u[ii]*(du[ii-2] - p.c[i])
    end
end
"""
function roessler_network(::Type{Val{:jac}},J,u,p::roessler_parameters,t)
    for i=1:p.N
        ii=3*i
        for j=1:p.N
            jj=3*i
            if i==j
                J[ii-2,jj-2] = -p.K*p.L[i,j]
                J[ii-2,jj-1] = -1.
                J[ii-2,jj] = -1.
                J[ii-1,jj-2] = 1.
                J[ii-1,jj-1] = p.a[i]
                J[ii-1,jj] = 0
                J[ii,jj-2] = u[ii]
                J[ii,jj-1] = 0
                J[ii,jj] = u[ii-2] - p.c[i]
            else
                J[ii-2,jj-2] = -p.K*p.L[i,j]
                J[ii-2,jj-1] = 0
                J[ii-2,jj] = 0
                J[ii-1,jj-2] = 0
                J[ii-1,jj-1] = 0
                J[ii-1,jj] = 0
                J[ii,jj-2] = 0
                J[ii,jj-1] = 0
                J[ii,jj] = 0
            end
        end
    end
    nothing
end
"""

@with_kw struct lotka_volterra_parameters <: DEParameters
    a::Matrix
    b::AbstractArray
    N::Int
end

function lotka_volterra(du, u, p::lotka_volterra_parameters, t)
    for i=1:p.N
        du[i] = 1.
        for j=1:p.N
            du[i] -= p.a[i,j]*p.N
        end
        du[i] *= p.b[i]*p.N
    end
end

###########
##### ODE Evaluation functions
###########

# define a custom ODE Problem type, so that we can also define a custom solve for it!
abstract type myMCProblem end

struct EqMCProblem <: myMCProblem
    p::MonteCarloProblem
    N_mc::Int64
    rel_transient_time::Float64 # float [0,1] (relative) time after which the trajectory/solution is saved and evaluated
end
# define structs for maps and custom solve based on dynamical systems library or discrete Problem

struct myMCSol
    sol::MonteCarloSolution
    N_mc::Int   # number of solutions saved / Monte Carlo trials runs
    N_t::Int  # number of time steps for each solutions
    N_meas::Int # number of measures used
    mc_prob::EqMCProblem # part of the struct to make saving/loading easier
end


# the type of problem that we are most interested: varying the combined initial conditions (ic) and parameter (par) space
# this routine helps setting up these problems
#   prob :: ODEProblem that defines the problem (ics and pars are irrelevant)
#   ic_ranges :: array with ranges (eg. 0:0.1:10) that ics are varyied. in same order as in the definition of the dynamical system. (AbstractArray are both Arrays and Ranges!)
#   parameters :: parameter struct of the ODE
#   var_par :: tuple of the field name of the parameter (as symbol) to vary and the range in which it should vary, e.g. (:K, 0:0.1:1)
#
# returns:
#   ic_par_problem :: function mapping (prob, i, repeat) tuple to new ODEProblem, needed by MonteCarloProblem
#   ic_par :: N_mc x N_dim sized array that holds the values of the ICs and parameters for each Iteration
#   N_mc :: int, number of ODEProblems to solve, needed for solve()
function setup_ic_par_mc_problem(prob::ODEProblem, ic_ranges::Array{T,1}, parameters::DEParameters, var_par::Tuple{Symbol,AbstractArray}) where T <: AbstractArray
    N_dim_ic = length(ic_ranges)
    N_dim = N_dim_ic + 1

    # construct a 2d-array that holds all the ICs and Parameters for the MonteCarlo run
    (ic_par, N_mc) = _ic_par_matrix(N_dim_ic, N_dim, ic_ranges, var_par)
    ic_par_problem = (prob, i, repeat) -> ODEProblem(prob.f, ic_par[i,1:N_dim_ic], prob.tspan, reconstruct(parameters; (var_par[1], ic_par[i,N_dim])))
    (ic_par_problem, ic_par, N_mc)
end

# TO-DO: one could probably combine both methods by using remake()
function setup_ic_par_mc_problem(prob::DiscreteProblem, ic_ranges::Array{T,1}, parameters::DEParameters, var_par::Tuple{Symbol,AbstractArray}) where T <: AbstractArray
    N_dim_ic = length(ic_ranges)
    N_dim = N_dim_ic + 1
    (ic_par, N_mc) = _ic_par_matrix(N_dim_ic, N_dim, ic_ranges, var_par)
    ic_par_problem = (prob, i, repeat) -> DiscreteProblem(prob.f, ic_par[i,1:N_dim_ic], prob.tspan, reconstruct(parameters; (var_par[1], ic_par[i,N_dim])))
    (ic_par_problem, ic_par, N_mc)
end

function setup_ic_par_mc_problem(prob::DEProblem, ic_gens::Array{T,1}, N_ic::Int, parameters::DEParameters, var_par::Tuple{Symbol,AbstractArray}) where T <: Function
    N_dim_ic = length(prob.u0)
    N_dim = N_dim_ic + 1
    (ic_par, N_mc) = _ic_par_matrix(N_dim_ic, N_dim, N_ic, ic_gens, var_par)
    #ic_par_problem = (prob, i, repeat) -> ODEProblem(prob.f, ic_par[i,1:N_dim_ic], prob.tspan, reconstruct(parameters; (var_par[1], ic_par[i,N_dim])))
    ic_par_problem = define_new_problem(prob, ic_par, parameters, N_dim_ic, ic_gens, var_par)
    (ic_par_problem, ic_par, N_mc)
end

# functions defining new problems that generate new ics when the trial needs to be repeated
function define_new_problem(prob::ODEProblem, ic_par::AbstractArray, parameters::DEParameters, N_dim_ic::Int, ic_gens::Array{T,1}, var_par::Tuple{Symbol,AbstractArray}) where T <: Function
    function new_problem(prob, i, repeat)
        if repeat > 1
            if repeat > 10
                error("More than 10 Repeats of a Problem in the Monte Carlo Run, there might me something wrong here!")
            else
                ic_par[i,1:N_dim_ic] = _new_ics(N_dim_ic,ic_gens)
            end
        end
        ODEProblem(prob.f, ic_par[i,1:N_dim_ic], prob.tspan,  reconstruct(parameters; (var_par[1], ic_par[i,N_dim_ic+1])))
    end
    new_problem
end

# same but for Discrete Problems
# TO-DO: one could probably combine both methods by using remake()
function define_new_problem(prob::DiscreteProblem, ic_par::AbstractArray, parameters::DEParameters, N_dim_ic::Int, ic_gens::Array{T,1}, var_par::Tuple{Symbol,AbstractArray}) where T <: Function
    function new_problem(prob, i, repeat)
        if repeat > 1
            if repeat > 10
                println("------------------")
                println("Error with IC/Par:")
                println(ic_par[i,:])
                println("------------------")
                error("More than 10 Repeats of a Problem in the Monte Carlo Run, there might me something wrong here!")
            else
                ic_par[i,1:N_dim_ic] = _new_ics(N_dim_ic,ic_gens)
            end
        end
        DiscreteProblem(prob.f, ic_par[i,1:N_dim_ic], prob.tspan,  reconstruct(parameters; (var_par[1], ic_par[i,N_dim_ic+1])))
    end
    new_problem
end


# helper function for setup_ic_par_mc_problem()
# uses ranges for the initial cond.
function _ic_par_matrix(N_dim_ic::Int, N_dim::Int, ic_ranges::Array{T,1}, var_par::Tuple{Symbol,AbstractArray}) where T <: AbstractArray
    N_ic_pars = zeros(Int, N_dim)
    for (i_range, ic_range) in enumerate(ic_ranges)
        N_ic_pars[i_range] = length(collect(ic_range))
    end
    N_ic_pars[N_dim] = length(collect(var_par[2]))
    N_mc = prod(N_ic_pars)
    N_ic_pars = tuple(N_ic_pars...) # need this as tuple for CartesianRange

    ic_par = zeros((N_mc, N_dim))
    for (i_mc, i_ci) in enumerate(CartesianRange(N_ic_pars))
         for i_dim=1:N_dim_ic
             ic_par[i_mc, i_dim] = ic_ranges[i_dim][i_ci[i_dim]]
         end
         ic_par[i_mc, N_dim] = var_par[2][i_ci[N_dim]]
    end
    (ic_par, N_mc)
end

# helper function for setup_ic_par_mc_problem()
# uses (random) generator functions for the initial cond.
function _ic_par_matrix(N_dim_ic::Int, N_dim::Int, N_ic::Int, ic_gens::Array{T,1},  var_par::Tuple{Symbol,AbstractArray}) where T<:Function

    N_ic_pars = (N_ic, length(collect(var_par[2])))
    N_mc = prod(N_ic_pars)
    N_gens = length(ic_gens)
    if N_dim_ic % N_gens != 0
        err("Number of initial cond. genators and Number of initial cond. doesn't fit together")
    end
    N_gen_steps = Int(N_dim_ic / N_gens)
    ics = zeros((N_ic, N_dim_ic))
    for i_ic=1:N_ic # loop over ICs
        for i_gen_steps=1:N_gen_steps  # loops over phase space dim
            for i_gen=1:N_gens
                ics[i_ic, N_gens*i_gen_steps - (N_gens - i_gen)] = ic_gens[i_gen]() # not like this!
            end
        end
    end

    ic_par = zeros((N_mc, N_dim))
    for (i_mc, i_ci) in enumerate(CartesianRange(N_ic_pars))
        ic_par[i_mc, 1:N_dim_ic] = ics[i_ci[1],:]
        ic_par[i_mc, end] = var_par[2][i_ci[2]]
    end
    (ic_par, N_mc)
end

# helper functions, calculate new ICs in case the MonteCarlo trial needs to be repeated
function _new_ics(N_dim_ic::Int, ic_gens::Array{T,1}) where T<:Function
    N_gens = length(ic_gens)
    N_gen_steps = Int(N_dim_ic / N_gens)
    ics = zeros(N_dim_ic)
    for i_gen_steps=1:N_gen_steps
        for i_gen=1:N_gens
            ics[N_gens*i_gen_steps - (N_gens - i_gen)] = ic_gens[i_gen]()
        end
    end
    ics
end


# custom solve for the EqMCProblem defined earlier. solves the MonteCarlo Problem for OrdinaryDiffEq, but saves and evaluates only the transient at a constant step size
# prob :: MC Problem of type defined in this library
function solve(prob::EqMCProblem, alg=nothing, N_t=400::Int, kwargs...)
    t_save = collect(tsave_array(prob.p.prob, N_t, prob.rel_transient_time))
    if alg!=nothing
        sol = solve(prob.p, alg, num_monte=prob.N_mc, dense=false, save_everystep=false, saveat=t_save, savestart=false, parallel_type=:parfor; kwargs...)
    else
        sol = solve(prob.p, num_monte=prob.N_mc, dense=false, save_everystep=false, saveat=t_save, savestart=false, parallel_type=:parfor; kwargs...)
    end
    mysol = myMCSol(sol, prob.N_mc, N_t, length(sol[1]), prob)

    inf_nan = check_inf_nan(mysol)
    if (length(inf_nan["Inf"])>0) | (length(inf_nan["NaN"])>0)
        warn("Some elements of the solution are Inf or NaN, check the solution with check_inf_nan again!")
    end
    mysol
end


# tsave_array
# given a tspan to be used in a ODEProblem, returns the array/iterator of time points to be saved (saveat argument of solve())
# tspan :: tspan also used in a ODEProblem
# N_t :: number of time points to be saved
# rel_transient_time :: only time points after the transient time are saved
function tsave_array(prob::ODEProblem, N_t::Int, rel_transient_time::Float64=0.7)
    tspan = prob.tspan
    t_tot_saved = (tspan[2] - tspan[1]) * (1 - rel_transient_time)
    t_start = tspan[2] - t_tot_saved
    dt = t_tot_saved / N_t
    t_start:dt:(tspan[2] - dt)
end

# Ignores N_t, because time steps needs to be Integer for Discrete Problems
function tsave_array(prob::DiscreteProblem, N_t::Int, rel_transient_time::Float64=0.7)
    tspan = prob.tspan
    t_tot_saved = Int(round((tspan[2] - tspan[1]) * (1 - rel_transient_time)))
    t_start = tspan[2] - t_tot_saved
    t_start:1:(tspan[2] - 1)
end

# eval_ode_run, (sol, i) -> (evaluated_solution, repeat=False)
# evaluates each ODE run and computes all the statistics needed for the further calculations
# right now these are: mean, std, skewness, relative entropy / KL div. to a gaussian and curve entropy
#
# input:
# sol :: result of one of the monte carlo ode run, should have only timesteps with constant time intervals between them
# i :: Int, number of iteration
# state_filter :: array with indicies of all dimensions that should be evaluated
function eval_ode_run(sol, i, state_filter::Array{Int64,1})

    (N_dim, N_t) = size(sol)

    m = zeros(Float64, N_dim)
    std = zeros(Float64, N_dim)
    #skew = zeros(Float64, N_dim)
    kl = zeros(Float64, N_dim)

    for i_dim in state_filter
        sol_i = sol[i_dim,2:end] # with the saveat solver option, tspan[1] is always saved as well but is not part of the transient.
        (m[i_dim],std[i_dim]) = StatsBase.mean_and_std(sol_i; corrected=true)
        #skew[i_dim] = StatsBase.skewness(sol_i, m[i_dim])

        # if the std is zero or extremly close to zero (meaning all values are the same, thus a fix point), the KL divergence relative to a Gaussian of same mean and std is hard to numerically compute.
        # we set it to zero in this case because both the empirical and the reference distribtuion are in this case approximately delta distributions at the same locaction, thus identical, having KL=0
        if std[i_dim] < 1e-10
            kl[i_dim] = 0
        else
            #ref_dist = Distributions.Normal(m[i_dim], std[i_dim]) # old
            kl[i_dim] = empirical_1D_KL_divergence(sol_i, m[i_dim], std[i_dim])
        end
    end

    # for test purposes we first write the same N-dimensonal curve entropy in every dimension
    ce = ones(N_dim)*curve_entropy(sol.u[2:end])
    ((m, std, kl, ce), false)
end

# MonteCarloProblem needs a function with only (sol, i) as inputs and this way the default of all dimensions beeing evaluated is easier to handle than with an optional/keyword argument
function eval_ode_run(sol, i)
    (N_dim, __) = size(sol)
    state_filter = collect(1:N_dim)
    eval_ode_run(sol, i, state_filter)
end

# include a return code check and repeat if integration failed. if this is not used with randomly generated initial conditions, this could leed into an endless loop!
function eval_ode_run_repeat(sol, i)
    if (sol.retcode != :Success) & (sol.retcode != :Default)
        return ((),true)
    end
    eval_ode_run(sol, i )
end

# include a return code check and set all results to Inf is the integration fails due to one or more variables exploding towards infitiy
function eval_ode_run_inf(sol, i)
    if (sol.retcode == :DtLessThanMin)
        last = sol.u[end]
        N_dim = length(last)
        inf_flag = false
        for i=1:N_dim
            if abs(last[i]) > 1e12
                inf_flag = true
            end
        end
        return ((ones(N_dim).*Inf,ones(N_dim).*Inf,ones(N_dim).*Inf,ones(N_dim).*Inf),false)
    end
    eval_ode_run(sol,i)
end

# checks if any of the results is Inf or NaN and returns the indices in a dictionary
function check_inf_nan(sol::myMCSol)
    N = sol.N_mc
    N_meas = sol.N_meas
    nan_inf = Dict("Inf" => ([]), "NaN" => ([]))
    for i=1:N
        for i_meas=1:N_meas
            if sum(isnan.(sol.sol[i][i_meas])) > 0
                push!(nan_inf["NaN"], [i,i_meas])
            end
            if sum(isinf.(sol.sol[i][i_meas])) > 0
                push!(nan_inf["Inf"], [i,i_meas])
            end
        end
    end
    nan_inf
end

# empirical_1D_KL_divergence
# NOT USED by default (instead the Perez-Cruz estimate below is used)
#
# based on histograms (might actually not be a good estimator for KL)
# u :: Input Array
# reference pdf: e.g. Normal(mean,std)
# hist_bins: number of bins of the histogram to estimate the empirical pdf of the data
function empirical_1D_KL_divergence(u::AbstractArray, reference_pdf::Distributions.UnivariateDistribution, hist_bins::Int)
    hist = StatsBase.fit(StatsBase.Histogram, u; closed=:left, nbins=hist_bins)
    hist = StatsBase.normalize(hist)
    bin_centers = @. hist.edges[1] + 0.5*(hist.edges[1][2] - hist.edges[1][1])
    refpdf_discrete = Distributions.pdf.(reference_pdf, bin_centers[1:end-1])

    StatsBase.kldivergence(hist.weights, refpdf_discrete)
end

# KL divergence
# estimate based on Perez-Cruz (IEEE, 2008)
# estimates the KL divergence by using linearly interpolated empirical CDFs.
# TO-DO: for large time series (N>10000) there is small risk that this yields Inf because the spacing between the samples becomes so small that the precision is not high enough to yield finite numbers.
##
function empirical_1D_KL_divergence(u::AbstractArray, mu::Number, sig::Number)

    N = length(u)
    Us = sort(u)
    Us_u = unique(Us)
    if length(Us_u)==1
        return 0.
    end

    eps = BigFloat(0.5 * minimum(diff(Us_u))) # we need very high precision for many values to not underflow to 0.

    # here we sample from the reference distribution. # OLD VERSION
    #samp = rand(reference_pdf, 10000)
    #samps = sort(samp)
    #samps_u = unique(samps)
    #eps2 = BigFloat(0.5 * minimum(diff(samps_u)))
    #if eps2 < eps
    #    eps = eps2
    #end

    ecdf_u = ecdf_pc(Us, Us_u, eps)
    dpc(x::BigFloat) = ecdf_u[x] - ecdf_u[x - eps]

    #ecdf_samp = ecdf_pc(samps, samps_u, eps2) # OLD VERSION
    #dqc(x::Real) =  ecdf_samp[x] - ecdf_samp[x - eps]

    normal_cdf(x::BigFloat) = BigFloat(0.5)*(BigFloat(1.)+erf((x-BigFloat(mu))/sqrt(2*BigFloat(sig)*BigFloat(sig))))
    dqc(x::BigFloat) = normal_cdf(x) - normal_cdf(x-eps)

    kld::Float64 = 0
    for i=1:N
        dp = dpc(BigFloat(Us[i]))
        dq = dqc(BigFloat(Us[i]))
        if (dp==0.) & (dq > 0.)
            kld += 0
        elseif (dp > 0.) & (dq > 0.)
            kld += log(dp/dq)
        else
            kld += Inf
        end
    end
    kld *= 1./N
    kld -= 1    # Perez-Cruz estimator converges as KL_pc - 1 -> KL
end

#
# Empirical Cumulative Densitiy function for KL divergence (w/ Heaviside(0)=1/2 and linear approx. between the points) based on Perez-Cruz, IEEE, 2008
# assumes that X is sorted and Xu is unique(X)!
function ecdf_pc(X::AbstractArray, Xu::AbstractArray, eps::BigFloat)
    N = length(X)
    Nu = length(Xu)
    ecdf = (0.5:1:(N-0.5))/N # 0.5 because Heaviside(0)=1/2 in this defintion
    eps = 0.05*eps
    dat_range = 2*(X[end] - X[1])

    # interpolation won't work with doublicate values and some extreme cases cases like only 2 or 3 unique values / delta peaks need extra care
    # the aim is to add additonal values at xi - eps, just a little bit infront of the delta peaks, to get a good cdf estimate

    # if all values are unique this is not necessary
    if Nu < N
        iu = [] # indices of unique elements, including the last element of series of non-unique elements
        i_fnu = [] # indices of the first element of series of non-unique elements

        if X[1]==X[2]
            push!(i_fnu,1)
        end
        for i=2:(N-1)
            if X[i-1]!=X[i]
                push!(iu, i-1)
                if X[i]==X[i+1]
                    push!(i_fnu, i)
                end
            end
        end
        if X[N-1]!=X[N]
            push!(iu, Nu-1)
        end
        push!(iu, N)
        x_fnu = X[i_fnu] .- eps
        ecdf_fnu = ecdf[i_fnu]

        # we also add 0 and 1 for a good asymptotic Behaviour
        # this way of array construction is quite slow, could replace with something that performs better instead

        itp = interpolate((sort([X[1]-2*dat_range,Xu...,x_fnu...,X[end]+2*dat_range]),), sort([0,ecdf[iu]...,ecdf[i_fnu]...,1]), Gridded(Linear()))
    else
        itp = interpolate((X,), ecdf, Gridded(Linear()))
    end
    itp
end
ecdf_pc(X::AbstractArray) = ecdf_pc(X, unique(X), 0.5*minimum(diff(unique(X))))
ecdf_pc(X::AbstractArray, Xu::AbstractArray, eps::Float64) = ecdf_pc(X, unique(X), BigFloat(0.5*minimum(diff(unique(X)))))

# curve entropy according to Balestrino et al, 2009, Entropy Journal
# could be used as an additional measure for the clustering
# bounded [0,1]
#
function curve_entropy(u::Array{Array{Float64,1},1}, r_eps::Float64=1e-15)
    D = mcs_diameter(u)
    if D > r_eps
        ce = log(curve_length(u)/D)/log(length(u) - 1)
    else
        ce = 0. # if the curve is just a point, its entropy is 0 (the logarithm would yield NaN)
    end
    ce
end

function curve_length(u::Array{Array{Float64,1},1})
    L::Float64 = 0.
    for it=2:size(u)[1]
        L += euclidean(u[it],u[it-1])
    end
    L
end

# Minimal Covering (Hyper)sphere of the points of the Curve
function mcs_diameter(u::Array{Array{Float64,1},1})
    # miniball routine needs a N_t x d matrix
    mcs = miniball(transpose(hcat(u...)))
    2.*sqrt(mcs.squared_radius)
end

###########
##### Results Evaluation functions
###########

# compute the distance matrix used for the dbSCAN clustering. here, we could experiment with different ways how to setup this matrix
# TO-DO: use symmetry: optimize for less memory consuption and computation time
function distance_matrix(sol::myMCSol, distance_func::Function)
    D = zeros((sol.N_mc, sol.N_mc))
    for i=1:sol.N_mc
        for j=1:sol.N_mc
            D[i,j] = distance_func(sol.sol.u[i], sol.sol.u[j])
        end
    end
    D
end
distance_matrix(sol::myMCSol) = distance_matrix(sol, weighted_norm)

# calculated the weighted norm between two trajectories, so one entry of the distance matrix
# x, y :: Tuples or Arrays containing all measures of the trajectories (e.g. means, vars per spatial dimension)
#
function weighted_norm(x, y, norm_function::Function, weights::AbstractArray=[1., 0.5, 0.5, 0.25])
    N_dim_meas::Int64 = length(x)
    out::Float64 = 0
    for i_dim=1:N_dim_meas
        out += weights[i_dim]*norm_function(x[i_dim] .- y[i_dim])
    end
    out
end
# use l1-norm by default
weighted_norm(x, y, weights::AbstractArray=[1., 0.5, 0.5, 0.25, 0.125]) = weighted_norm(x,y, in -> sum(abs.(in)), weights)

function cluster()
    false
end

# return mean values of all measures for each cluster
function cluster_measures(sol::myMCSol, clusters::DbscanResult)
    N_cluster = length(clusters.seeds)
    N_dim = length(sol.sol.u[1][1])
    mean_measures = zeros((N_clusters,sol.N_meas,N_dim))
    for i_sol=1:sol.N_mc
        for i_meas=1:sol.N_meas
            mean_measures[clusters.assignments[i_sol],i_meas,:] += sol.sol.u[i_sol][i_meas]
        end
    end
    mean_measures ./ N_mc
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

end
