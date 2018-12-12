###########
##### Setup Monte Carlo Problem functions
###########
using DifferentialEquations
import DifferentialEquations.solve # this needs to be directly importet in order to extend it with our own solve() for our own problem struct
using Parameters
import Base.sort, Base.sort!
import LinearAlgebra.normalize


# Parameter Variation Struct
# This struct holds information about the parameters and how they chould
# be varied
# For many cases this struct is automaticlly initialized when
# calling BifAnaMCProblem with appropiate Tuples
#
# name, is the symbol of the name of the parameter to be varied
# new_val, is the function that determines to which value it is changed, takes the
#           index of the trial as an argument (i) -> new_value
# function that returns a new parameter when given the old parameter and the new
# value as a kwargs: (old_par; (Dict(name=>new_value))) -> new_par
struct ParameterVar
    name::Symbol
    new_val::Function
    new_par::Function

    function ParameterVar(name::Symbol, func::Function, new_par::Function)
        new(name,func,new_par)
    end

    function ParameterVar(name::Symbol, arr::AbstractArray, new_par::Function)
        function new_val(i)
            arr[i]
        end
        new(name,new_val,new_par)
    end
end
ParameterVar(name::Symbol, func::Function) = ParameterVar(name, func, reconstruct)
ParameterVar(name::Symbol, arr::AbstractArray) = ParameterVar(name, arr, reconstruct)



# define a custom ODE Problem type, so that we can also define a custom solve for it!

# The Main Scruct, defining a new Differential Equation Problem Type with its own solve()
# It defines a MonteCarloProblem over a initial condition - parameter space and only evaluates the tail of each trajectory
# Usage:
# p - Basic Differential Equations Problem [DiscreteProblem, ODEProblem, SDEProblem] are supported so far
# ic_gens - Array of Functions or Array of Numbers/Ranges, sets or generates the initial cond per dimension
# N_ic - if ic_gens is a array of functions / ICs generators, this is the total number of ICs that should be generated, if ic_gens is an Array of AbstractArray this argument is omited and not included in the function call
# pars - Parameter Instance of the Problem
# par_range_tuple - tuple with Symbol that is the name of the Parameter that should be varied and range or functions that governs how the Parameter is varied, an extra function can be given in case new parameters should be constructed with another function than the default reconstruct from Parameters.jl
# eval_ode_func - evalalution function for the MonteCarloProblem
# tail_frac - float [0,1] (relative) time after which the trajectory/solution is saved and evaluated, default value 0.9
#
struct BifAnaMCProblem <: myMCProblem
    p::MonteCarloProblem
    N_mc::Int64
    rel_transient_time::Float64 # float [0,1] (relative) time after which the trajectory/solution is saved and evaluated
    ic_par::AbstractArray # matrix that stores all ICs and Pars for each run
    par_var::Union{Tuple{Symbol,Union{AbstractArray,Function},<:Function},Tuple{Symbol,Union{AbstractArray,Function}}} # parameter variation tuple, is the same as par_range_tuple

    # inner constructer used for randomized ICs
    function BifAnaMCProblem(p::Union{DiscreteProblem, ODEProblem, SDEProblem}, ic_gens::Array{<:Function,1}, N_ic::Int, pars::DEParameters, par_range_tuple::Union{Tuple{Symbol,Union{AbstractArray,Function},<:Function},Tuple{Symbol,Union{AbstractArray,Function}}}, eval_ode_func::Function, tail_frac::Number)
        (ic_coupling_problem, ic_par, N_mc) = setup_ic_par_mc_problem(p, ic_gens, N_ic, pars, par_range_tuple)
        mcp = MonteCarloProblem(p, prob_func=ic_coupling_problem, output_func=eval_ode_func)
        new(mcp, N_mc, tail_frac, ic_par, par_range_tuple)
    end

    # inner constructer used for non-randomized ICs
    function BifAnaMCProblem(p::Union{DiscreteProblem, ODEProblem, SDEProblem}, ic_ranges::Array{<:AbstractArray,1}, pars::DEParameters, par_range_tuple::Union{Tuple{Symbol,Union{AbstractArray,Function},<:Function},Tuple{Symbol,Union{AbstractArray,Function}}}, eval_ode_func::Function, tail_frac::Number)
        (ic_coupling_problem, ic_par, N_mc) = setup_ic_par_mc_problem(p, ic_ranges, pars, par_range_tuple)
        mcp = MonteCarloProblem(p, prob_func=ic_coupling_problem, output_func=eval_ode_func)
        new(mcp, N_mc, tail_frac, ic_par, par_range_tuple)
    end

    # Direct Constructor
    BifAnaMCProblem(p::MonteCarloProblem, N_mc::Int64, rel_transient_time::Float64, ic_par::AbstractArray, par_range_tuple::Union{Tuple{Symbol,Union{AbstractArray,Function},<:Function},Tuple{Symbol,Union{AbstractArray,Function}}}) = new(p, N_mc, rel_transient_time, ic_par, par_range_tuple)
end
BifAnaMCProblem(p::Union{DiscreteProblem, ODEProblem, SDEProblem}, ic_gens::Function, N_ic::Int, pars::DEParameters, par_range_tuple::Union{Tuple{Symbol,Union{AbstractArray,Function},<:Function},Tuple{Symbol,Union{AbstractArray,Function}}}, eval_ode_func::Function, tail_frac::Number) = BifAnaMCProblem(p, [ic_gens], N_ic, pars, par_range_tuple, eval_ode_func, tail_frac)
BifAnaMCProblem(p::Union{DiscreteProblem, ODEProblem, SDEProblem}, ic_gens::Union{Array{<:Function,1},Function}, N_ic::Int, pars::DEParameters, par_range_tuple::Union{Tuple{Symbol,Union{AbstractArray,Function},<:Function},Tuple{Symbol,Union{AbstractArray,Function}}}, eval_ode_func::Function) = BifAnaMCProblem(p,ic_gens,N_ic,pars,par_range_tuple,eval_ode_func, 0.9)

# utility function that returns the parameter of a BifAnaMCProblem
parameter(p::BifAnaMCProblem) = p.ic_par[:,end]


# define structs for maps and custom solve based on dynamical systems library or discrete Problem

struct BifMCSol <: myMCSol
    sol::MonteCarloSolution
    N_mc::Int   # number of solutions saved / Monte Carlo trials runs
    N_t::Int  # number of time steps for each solutions
    N_dim::Int # sytem dimension
    N_meas::Int # number of measures used, N_meas = N_meas_dim + N_meas_global
    N_meas_dim::Int # number of measures that are evaluated for every dimension
    N_meas_global::Int # number of global measures, in the case of system dimension == 1, N_meas_global = 1 = N_meas and N_meas_dim = 0
end

# if using random ICs/Pars the solutions are also in random order. These functions returns the MonteCarloSolution object sorted by the value of the parameter
function sort(sol::BifMCSol, prob::BifAnaMCProblem)
    sol_copy = deepcopy(sol)
    prob_copy = deepcopy(prob)
    sort!(sol_copy, prob)
    sort!(prob_copy)
    (sol_copy, prob_copy)
end

function sort!(sol::BifMCSol, prob::BifAnaMCProblem)
    p = parameter(prob)
    sortind = sortperm(p)
    prob.ic_par[:,:] = prob.ic_par[sortind,:]
    sol.sol[:] = sol.sol[sortind]
end

function sort(prob::BifAnaMCProblem)
    prob_copy = deepcopy(prob)
    sort!(prob_copy)
    prob_copy
end

function sort!(prob::BifAnaMCProblem)
    sortind = sortperm(parameter(prob))
    prob.ic_par[:,:] = prob.ic_par[sortind,:]
end

# shows the results for specified parameter values between min_par and max_par
function show_results(sol::BifMCSol, prob::BifAnaMCProblem, min_par::Number, max_par::Number, sorted::Bool=false)
    if sorted==false
        (ssol, sprob) = sort(sol, prob)
    else
        ssol = sol
        sprob = prob
    end

    p = parameter(sprob)
    ssol.sol.u[(p .> min_par) .& (p .< max_par)]
end

# return the results for the k-th measure as an array
function get_measure(sol::BifMCSol, k::Int)
    if k <=  sol.N_meas_dim
        arr = zeros((sol.N_mc,sol.N_dim))
        for i=1:sol.N_mc
            arr[i,:] = sol.sol[i][k]
        end
    else
        arr = zeros(sol.N_mc)
        for i=1:sol.N_mc
            arr[i] = sol.sol[i][k]
        end
    end
    arr
end

# returns an instance of BifMCSol with the solutions normalized to be in range [0,1]
# it is possible to only select that some of the measures are normalized by providing an array with the indices of the measures
# that should be normalized, e.g. [1,2] for measure 1 and measure 2 to be normalized
function normalize(sol::BifMCSol, k::AbstractArray)
    N_meas = length(k)
    max_meas = zeros(sol.N_meas)
    min_meas = zeros(sol.N_meas)
    meas_ranges = zeros(sol.N_meas)
    for i in k
        meas_tmp = get_measure(sol, i)
        max_meas[i] = maximum(meas_tmp)
        min_meas[i] = minimum(meas_tmp)
    end
    meas_ranges = max_meas .- min_meas

    new_mc_sol = deepcopy(sol.sol)
    for i_meas in k
        for i_mc=1:sol.N_mc
            new_mc_sol.u[i_mc][i_meas][:] = (sol.sol.u[i_mc][i_meas][:] .- min_meas[i_meas]) ./ meas_ranges[i_meas]
        end
    end

    sol_new = BifMCSol(new_mc_sol, sol.N_mc, sol.N_t, sol.N_dim, sol.N_meas, sol.N_meas_dim, sol.N_meas_global)
end
normalize(sol::BifMCSol) = normalize(sol, 1:sol.N_meas)

# this routine replaces Infs with the specified values, this can under some circumstances be usefull when investigating systems that do not converge and you want to calculate distances

# the type of problem that we are most interested: varying the combined initial conditions (ic) and parameter (par) space
# this routine helps setting up these problems
#   prob :: ODEProblem that defines the problem (ics and pars are irrelevant)
#   ic_ranges :: array with ranges (eg. 0:0.1:10) that ics are varyied. in same order as in the definition of the dynamical system. (AbstractArray are both Arrays and Ranges!)
#   parameters :: parameter struct of the ODE
#   var_par :: tuple of the field name of the parameter (as symbol) to vary and the range in which it should vary, e.g. (:K, 0:0.1:1) and a function that returns a new set of parameters: the default is reconstruct from Parameters.jl / @with_kw, the function needs to be able to called like: var_par[3](old_par; (var_par[1],new_value)). This is possible e.g. with an inner constructer (with one keyword). See Kuramoto Chain as an example
#
# returns:
#   ic_par_problem :: function mapping (prob, i, repeat) tuple to new ODEProblem, needed by MonteCarloProblem
#   ic_par :: N_mc x N_dim sized array that holds the values of the ICs and parameters for each Iteration
#   N_mc :: int, number of ODEProblems to solve, needed for solve()
function setup_ic_par_mc_problem(prob::ODEProblem, ic_ranges::Array{T,1}, parameters::DEParameters, var_par::Union{Tuple{Symbol,Union{AbstractArray,Function},<:Function},Tuple{Symbol,Union{AbstractArray,Function}}}) where T <: AbstractArray
    N_dim_ic = length(ic_ranges)
    N_dim = N_dim_ic + 1
    var_par = _var_par_check(var_par)

    # construct a 2d-array that holds all the ICs and Parameters for the MonteCarlo run
    (ic_par, N_mc) = _ic_par_matrix(N_dim_ic, N_dim, ic_ranges, var_par)
    ic_par_problem = (prob, i, repeat) -> ODEProblem(prob.f, ic_par[i,1:N_dim_ic], prob.tspan, var_par[3](parameters; Dict(var_par[1]=>ic_par[i,N_dim])...))
    (ic_par_problem, ic_par, N_mc)
end

# for discrete problems and non-random ICs
function setup_ic_par_mc_problem(prob::DiscreteProblem, ic_ranges::Array{T,1}, parameters::DEParameters, var_par::Union{Tuple{Symbol,Union{AbstractArray,Function},<:Function},Tuple{Symbol,Union{AbstractArray,Function}}}) where T <: AbstractArray
    N_dim_ic = length(ic_ranges)
    N_dim = N_dim_ic + 1
    var_par = _var_par_check(var_par)

    (ic_par, N_mc) = _ic_par_matrix(N_dim_ic, N_dim, ic_ranges, var_par)
    ic_par_problem = (prob, i, repeat) -> DiscreteProblem(prob.f, ic_par[i,1:N_dim_ic], prob.tspan, var_par[3](parameters; Dict(var_par[1]=>ic_par[i,N_dim])...))
    (ic_par_problem, ic_par, N_mc)
end

# for sdeproblems and non-random ICs
function setup_ic_par_mc_problem(prob::SDEProblem, ic_ranges::Array{T,1}, parameters::DEParameters, var_par::Union{Tuple{Symbol,Union{AbstractArray,Function},<:Function},Tuple{Symbol,Union{AbstractArray,Function}}}) where T <: AbstractArray
    N_dim_ic = length(ic_ranges)
    N_dim = N_dim_ic + 1
    var_par = _var_par_check(var_par)

    (ic_par, N_mc) = _ic_par_matrix(N_dim_ic, N_dim, ic_ranges, var_par)
    ic_par_problem = (prob, i, repeat) -> SDEProblem(prob.f, prob.g, ic_par[i,1:N_dim_ic], prob.tspan, var_par[3](parameters; Dict(var_par[1]=> ic_par[i,N_dim])...))
    (ic_par_problem, ic_par, N_mc)
end

# for all problem types and random ICs
function setup_ic_par_mc_problem(prob::Union{DiscreteProblem, ODEProblem, SDEProblem}, ic_gens::Array{<:Function,1}, N_ic::Int, parameters::DEParameters, var_par::Union{Tuple{Symbol,Union{AbstractArray,Function},<:Function}, Tuple{Symbol,Union{AbstractArray,Function}}})
    N_dim_ic = length(prob.u0)
    N_dim = N_dim_ic + 1
    var_par = _var_par_check(var_par)

    (ic_par, N_mc) = _ic_par_matrix(N_dim_ic, N_dim, N_ic, ic_gens, var_par)
    ic_par_problem = define_new_problem(prob, ic_par, parameters, N_dim_ic, ic_gens, var_par)
    (ic_par_problem, ic_par, N_mc)
end
setup_ic_par_mc_problem(prob::Union{DiscreteProblem, ODEProblem, SDEProblem}, ic_gens::Function, N_ic::Int, parameters::DEParameters, var_par::Union{Tuple{Symbol,Union{AbstractArray,Function},<:Function},Tuple{Symbol,Union{AbstractArray,Function}}}) = setup_ic_par_mc_problem(prob, [ic_gens], N_ic, parameters, var_par)

function _var_par_check(var_par::Union{Tuple{Symbol,Union{AbstractArray,Function},<:Function}, Tuple{Symbol,Union{AbstractArray,Function}}})
    if length(var_par)==2
        new_var_par = (var_par[1],var_par[2],reconstruct)
        var_par = new_var_par
    else
        new_var_par = var_par
    end
    return new_var_par
end

# functions defining new problems that generate new ics when the trial needs to be repeated
function define_new_problem(prob::ODEProblem, ic_par::AbstractArray, parameters::DEParameters, N_dim_ic::Int, ic_gens::Array{T,1}, var_par::Tuple{Symbol,Union{AbstractArray,Function},<:Function}) where T <: Function
    function new_problem(prob, i, repeat)
        _repeat_check(repeat, ic_par, ic_gens)
        ODEProblem(prob.f, ic_par[i,1:N_dim_ic], prob.tspan,  var_par[3](parameters; Dict(var_par[1]=> ic_par[i,N_dim_ic+1])...))
    end
    new_problem
end

# same but for Discrete Problems
# TO-DO: one could probably combine both methods by using remake()
function define_new_problem(prob::DiscreteProblem, ic_par::AbstractArray, parameters::DEParameters, N_dim_ic::Int, ic_gens::Array{<:Function,1}, var_par::Tuple{Symbol,Union{AbstractArray,Function},<:Function})
    function new_problem(prob, i, repeat)
        _repeat_check(repeat, ic_par, ic_gens)
        DiscreteProblem(prob.f, ic_par[i,1:N_dim_ic], prob.tspan,  var_par[3](parameters; Dict(var_par[1] => ic_par[i,N_dim_ic+1])...))
    end
    new_problem
end

function define_new_problem(prob::SDEProblem, ic_par::AbstractArray, parameters::DEParameters, N_dim_ic::Int, ic_gens::Array{<:Function,1}, var_par::Tuple{Symbol,Union{AbstractArray,Function},<:Function})
    function new_problem(prob, i, repeat)
        _repeat_check(repeat, ic_par, ic_gens)
        SDEProblem(prob.f, prob.g, ic_par[i,1:N_dim_ic], prob.tspan,  var_par[3](parameters; Dict(var_par[1] => ic_par[i,N_dim_ic+1])...))
    end
    new_problem
end

# checks if the problem has to be repeated, if so, it generates new ICs
function _repeat_check(repeat, ic_par::AbstractArray, ic_gens::Array{<:Function,1})
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
end


# helper function for setup_ic_par_mc_problem(), sets up the big IC-parameter matrix for all runs
# uses ranges for the initial cond.
function _ic_par_matrix(N_dim_ic::Int, N_dim::Int, ic_ranges::Array{T,1}, var_par::Tuple{Symbol,AbstractArray,<:Function}) where T <: AbstractArray
    N_ic_pars = zeros(Int, N_dim)
    for (i_range, ic_range) in enumerate(ic_ranges)
        N_ic_pars[i_range] = length(collect(ic_range))
    end
    N_ic_pars[N_dim] = length(collect(var_par[2]))
    if prod(float(N_ic_pars)) > 1e10
        @warn "More than 1e10 initial cond. Are you sure what you are doing? Overflows might occur."
    end
    N_mc = prod(N_ic_pars)
    if N_mc==0
        error("Zero inital conditions. Either at least one of the ranges has length 0 or an overflow occured")
    end

    N_ic_pars = tuple(N_ic_pars...) # need this as tuple for CartesianIndices

    ic_par = zeros((N_mc, N_dim))
    for (i_mc, i_ci) in enumerate(CartesianIndices(N_ic_pars))
         for i_dim=1:N_dim_ic
             ic_par[i_mc, i_dim] = ic_ranges[i_dim][i_ci[i_dim]]
         end
         ic_par[i_mc, N_dim] = var_par[2][i_ci[N_dim]]
    end
    (ic_par, N_mc)
end

# helper function for setup_ic_par_mc_problem()
# uses (random) generator functions for the initial cond. AND the parameter evaluated_solution
function _ic_par_matrix(N_dim_ic::Int, N_dim::Int, N_ic::Int, ic_gens::Array{<:Function,1},  var_par::Tuple{Symbol,Function,<:Function})
    N_gens = length(ic_gens) # without the parameter geneartor
    if N_dim_ic % (N_gens) != 0
        err("Number of initial cond. genators and Number of initial cond. doesn't fit together")
    end
    N_gen_steps = Int(N_dim_ic / N_gens)
    ic_par = zeros((N_ic, N_dim))
    for i_ic=1:N_ic
        for i_gen_steps=1:N_gen_steps
            for i_gen=1:N_gens
                ic_par[i_ic, N_gens*i_gen_steps - (N_gens - i_gen)] = ic_gens[i_gen]()
            end
        end
        ic_par[i_ic,N_dim] = var_par[2]()
    end
    (ic_par, N_ic)
end


# helper function for setup_ic_par_mc_problem()
# uses (random) generator functions for the initial cond.
function _ic_par_matrix(N_dim_ic::Int, N_dim::Int, N_ic::Int, ic_gens::Array{T,1},  var_par::Tuple{Symbol,AbstractArray,<:Function}) where T<:Function
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
                ics[i_ic, N_gens*i_gen_steps - (N_gens - i_gen)] = ic_gens[i_gen]()
            end
        end
    end

    ic_par = zeros((N_mc, N_dim))
    for (i_mc, i_ci) in enumerate(CartesianIndices(N_ic_pars))
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


# custom solve for the BifAnaMCProblem defined earlier. solves the MonteCarlo Problem for OrdinaryDiffEq, but saves and evaluates only after transient at a constant step size, the results are sorted by parameter value
# prob :: MC Problem of type defined in this library
# alg :: Algorithm to use, same as for solve() from DifferentialEquations.jl
# N_t :: Number of timesteps to be saved
# parallel_type :: which form of parallelism should be used? same as for MonteCarloProblem from DifferentialEquations.jl
# flag_check_inf_nan :: Does a check if any of the results are NaN or inf
# custom_solve:: Function/Nothing, custom solve function
# euler_inf :: If true, solves it Euler(), constant stepsize and tstops, usefull for some easy, but instable test models
function solve(prob::BifAnaMCProblem, alg=nothing, N_t=400::Int, parallel_type=:parfor; flag_check_inf_nan=true, custom_solve::Union{Function,Nothing}=nothing, kwargs...)
    t_save = collect(tsave_array(prob.p.prob, N_t, prob.rel_transient_time))

    if custom_solve!=nothing
        sol = custom_solve(prob, t_save)
    elseif alg!=nothing
        sol = solve(prob.p, alg=alg, num_monte=prob.N_mc, dense=false, save_everystep=false, saveat=t_save, savestart=false, parallel_type=parallel_type; kwargs...)
    else
        sol = solve(prob.p, num_monte=prob.N_mc, dense=false, save_everystep=false, saveat=t_save, savestart=false, parallel_type=parallel_type; kwargs...)
    end


    """
    if alg!=nothing
        sol = solve(prob.p, alg, num_monte=prob.N_mc, dense=false, save_everystep=false, saveat=t_save, savestart=false, parallel_type=parallel_type; kwargs...)
    else
        sol = solve(prob.p, num_monte=prob.N_mc, dense=false, save_everystep=false, saveat=t_save, savestart=false, parallel_type=parallel_type; kwargs...)
    end
    """

    mysol = BifMCSol(sol, prob.N_mc, N_t, length(prob.p.prob.u0), get_measure_dimensions(sol)...)

    if flag_check_inf_nan
        inf_nan = check_inf_nan(mysol)
        if (length(inf_nan["Inf"])>0) | (length(inf_nan["NaN"])>0)
            @warn "Some elements of the solution are Inf or NaN, check the solution with check_inf_nan again!"
        end
    end
    sort!(mysol, prob)
    mysol
end
solve_euler_inf(prob::BifAnaMCProblem, t_save::AbstractArray; dt=0.1) = solve(prob.p, alg=Euler(), dt=dt, num_monte=prob.N_mc, parallel_type=:parfor, dense=false, saveat=t_save, tstops=t_save, savestart=false, save_everystep=false)

function get_measure_dimensions(sol)
    sol1 = sol[1] # use the first solution for the dimension determination

    N_meas_dim = 0
    N_meas_global = 0

    for i=1:length(sol1)
        if length(sol1[i]) > 1
            N_meas_dim += 1
        else
            N_meas_global += 1
        end
    end
    (N_meas_dim + N_meas_global, N_meas_dim, N_meas_global)
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
