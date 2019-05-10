###########
##### Setup Monte Carlo Problem functions
###########
using DifferentialEquations
import DifferentialEquations.solve # this needs to be directly importet in order to extend it with our own solve() for our own problem struct
using Parameters
import Base.sort, Base.sort!, Base.zero
import LinearAlgebra.normalize

"""
    ParameterVar

Parameter Variation types, these structs holds information about the parameters and how they should be varied. For many cases this struct is automaticlly initialized when calling `DEMCBBProblem` with appropiate Tuples. It assumes that the parameters are itself structs most commonly with a field that has to be varied.

# Type Hierachy
* `OneDimParameterVar, MultiDimParameterVar <: ParameterVar`
* `ParameterVarArray, ParameterVarFunc <: OneDimParameterVar`
* `MulitDimParameterVarArray, MultiDimParameterVarFunc <: MultiDimParameterVar`
"""
abstract type ParameterVar end

"""
    OneDimParameterVar

Paramter Variation types for setups with one parameter.

# Subtypes
* ['ParameterVarFunc'](@ref)
* ['ParameterVarFunc'](@ref)

# Common constructors
* `OneDimParameterVar(name::Symbol,new_val::AbstractArray,new_par::Function)``
* `OneDimParameterVar(name::Symbol,new_val::AbstractArray)`
* `OneDimParameterVar(name::Symbol,new_val::Function, new_par::Function)`
* `OneDimParameterVar(name::Symbol,new_val::Function)`

* `name`: Symbol of the name of the parameter
* `new_val`:: Function that returns a new value, signature: `(i::Int) -> new_value::Number` or `()-> new_value::Number` or Array with all parameter values
* `new_par`:: Function that returns a new parameter struct, default: Parameters.reconstruct, signature: `(old_par; Dict(name=>new_val)) = new_par`
"""
abstract type OneDimParameterVar <: ParameterVar end

"""
    ParameterVarArray

Type for the parameter variation with an array that holds all parameter values.
The struct has the fields:

* `name`: Symbol of the name of the parameter
* `new_val`:: Function that returns a new value, signature: `(i::Int) -> new_value::Number` or `()-> new_value::Number`
* `new_par`:: Function that returns a new parameter struct, default: Parameters.reconstruct, signature: `(old_par; Dict(name=>new_val)) = new_par`
* `N` :: Length of the array / Number of parameter values
* `arr` :: The input array saved

# Initialization
* `ParameterVar(name::Symbol, arr::AbstractArray, new_par::Function)`
* `ParameterVar(name::Symbol, arr::AbstractArray)`
* `name` is the name of the field of the parameter struct that should be varied
* `arr` is the array that containts all values the parameter should be varied to.
"""
struct ParameterVarArray <: OneDimParameterVar
    name::Symbol
    new_val::Function
    new_par::Function
    N::Integer
    arr::AbstractArray

    function ParameterVarArray(name::Symbol, arr::AbstractArray, new_par::Function)
        function new_val(i)
            arr[i]
        end
        new(name,new_val,new_par,length(collect(arr)),arr)
    end
end
ParameterVarArray(name::Symbol, arr::AbstractArray) = ParameterVarArray(name, arr, reconstruct)

"""
    ParameterVarFunc

Struct for the parameter variation with a function that generates new values.
The struct has the fields:

* `name`: Symbol of the name of the parameter
* `new_val`:: Function that returns a new value, signature: `(i::Int) -> new_value::Number`
* `new_par`:: Function that returns a new parameter struct, default: Parameters.reconstruct, signature: `(old_par; Dict(name=>new_val)) = new_par`

# Initialization
* `ParameterVar(name::Symbol, func::Function, new_par::Function)`
* `ParameterVar(name::Symbol, func::Function)`
* `name` is the name of the field of the parameter struct that should be varied
* `func` is the function that generates the parameter values, signature: `(i::Int) -> new_value::Number`
"""
struct ParameterVarFunc <: OneDimParameterVar
    name::Symbol
    new_val::Function
    new_par::Function

    function ParameterVarFunc(name::Symbol, func::Function, new_par::Function)
        func = verify_func(func)
        new(name,func,new_par)
    end
end
ParameterVarFunc(name::Symbol, func::Function) = ParameterVarFunc(name, func, reconstruct)

OneDimParameterVar(name::Symbol,new_val::AbstractArray,new_par::Function) = ParameterVarArray(name, new_val, new_par)
OneDimParameterVar(name::Symbol,new_val::AbstractArray) = ParameterVarArray(name, new_val)
OneDimParameterVar(name::Symbol,new_val::Function, new_par::Function) = ParameterVarFunc(name, new_val, new_par)
OneDimParameterVar(name::Symbol,new_val::Function) = ParameterVarFunc(name, new_val)

"""
    MultiDimParameterVar

Holds information about multiple parameters that should be varied simultaneously.
The struct has the fields:

* `data`: 1-D Array of `OneDimParamterVar`
* `Function`: function that returns a new parameter struct given keyword arguments of _all_ parameters that should be varied. signature: `(old_par; Dict(name_1=>new_val_1, name_2=>new_val_2, ...)) = new_par`
* `N`: Number of parameters that are varied.

Internally there are two different types, `MultiDimParameterVarFunc` and `MultiDimParameterVarArray`. The only difference is what type of ParameterVar they store. The different types are needed for dispatching on them in the routines that setups `DEMCBBProblem`

# Initialization
* `MultiDimParameterVar(data::Array{ParameterVarFunc,1}, func::Function)`
* `MultiDimParameterVar(data::Array{ParameterVarArray,1}, func::Function)`
* `MultiDimParameterVar(parvar::ParameterVar, func::Function)`
* `MultiDimParameterVar(parvar::ParameterVar)`: default function is Parameters.reconstruct
"""
abstract type MultiDimParameterVar <: ParameterVar end

struct MultiDimParameterVarFunc <: MultiDimParameterVar
    data::Array{ParameterVarFunc,1}
    new_par::Function
    N::Int
end

struct MultiDimParameterVarArray <: MultiDimParameterVar
    data::Array{ParameterVarArray,1}
    new_par::Function
    N::Int
end

MultiDimParameterVar(data::Array{ParameterVarFunc,1}, func::Function) = MultiDimParameterVarFunc(data, func, length(data))
MultiDimParameterVar(data::Array{ParameterVarArray,1}, func::Function) = MultiDimParameterVarArray(data, func, length(data))
MultiDimParameterVar(parvar::ParameterVar, func::Function) = MultiDimParameterVar([parvar], func)
MultiDimParameterVar(parvar::ParameterVar) = MultiDimParameterVar(parvar, reconstruct)

"""
    getindex(parvar::MultiDimParameterVar, i::Int)

Like regular arrays the individual ParameterVar entries can be accessed with square brackets e.g.: `parvar[i]`.
"""
Base.getindex(parvar::MultiDimParameterVar, i::Int) = parvar.data[i]

"""
    length(parvar::MultiDimParameterVar)

Length returns the amount of Parameters that are setup to be varied.
"""
Base.length(parvar::MultiDimParameterVar) = parvar.N
Base.length(par::OneDimParameterVar) = 1

"""
    ParameterVar(prob::myMCProblem)

Given one of the problem types of this library its ParameterVar is returned.
"""
ParameterVar(prob::myMCProblem) = prob.par_var

"""
    MCBBProblem <: myMCProblem

Abstract type of [`DEMCBBProblem`](@ref) (using DifferentialEquations.jl as a backend) and [`CustomMCBBProblem`](@ref) (using customized `solve` and problem functions).
"""
abstract type MCBBProblem <: myMCProblem end

"""
    DEMCBBProblem

Differential Equations Monte Carlo Basin Bifurcation Problem: Main type for the sample based bifurcation/stablity analysis based on `MonteCarloProblem` from DifferentialEquations. This struct holds information about the underlying differential equation and the parameters and initial conditions its supposed to be solved for. Many points from the initial conditions - parameter space are sampled. When solved the solutions is evaluated seperatly for each dimension and certain statistical measures like mean or standard deviation are saved.

The struct has several different constructors following below.

Note that its supertypes are `MCBBProblem` and `myMCProblem`, but not any of the DifferentialEquations abstract problem types.

The struct has the following fields:
* `p`: `MonteCarloProblem` to be solved, part of DifferentialEquations
* `N_mc`: Number of (Monte Carlo) runs to be solved
* `rel_transient_time`: Only after this time (relative to the total integration time) the solutions are evaluated
* `ic`: (``N_{mc} \\times (N_{dim_{ic}} + N_{par})``)-Matrix containing initial conditions for each run.
* `par`: (``N_{mc} \\times N_{par})-Matrix containitng parameter values for each run
* `par_var`: `ParameterVar`, information about how the parameters are varied, see [`ParameterVar`](@ref)

# Constructors

The type has two different main constructors and several others that do automatic type conversions of appropiate tuples to `ParamterVar` types or of functions and arrays to arrays of functions/arrays if needed.

## Randomized Initial conditions

    DEMCBBProblem(p::DiffEqBase.DEProblem, ic_gens::Array{<:Function,1}, N_ic::Int, pars::DEParameters, par_range_tuple::ParameterVar, eval_ode_func::Function, tail_frac::Number)

Setup a DEMCBBProblem with _randomized_ initial conditions (and parameters).

* `p`: A Problem from DifferentialEquations, currently supported are `DiscreteProblem`, `ODEProblem`, `SDEProblem`,`DDEProblem` the base problem one is interested in.
* `ic_gens`: A function or an array of functions that generate the initial conditions for each trial. Function signature is `()->new_value::Number` or `(i_run)->new_value::Number` or `()->new_value::AbstractArray` or `(i_run)->new_value::AbstractArray`. If only one function is provided its used for all IC dims, if ``M<N_{dim}`` functions with ``N_{dim}=k\\cdot M`` are provided these functions are repeated ``k`` times (usefull e.g. for coupled chaotic oscillators).
* `N_ic`: Number of trials to be computed, if parameter variation is varied by an array/range, `N_ic` is the number of initial conditions for each parameter value. Each parameter step then has the same `N_ic` idential initial conditions.
* `pars`: parameter struct of the underlying system
* `par_range_tuple`:  `ParameterVar`, information about how the parameters are varied, see [`ParameterVar`](@ref). Its also possible to hand over an appropiate tuple that will be automaticly converted to a `ParameterVar` type. A tuple of (First: name of the parameter as a symbol, Second: AbstractArray or Function that contains all parameters for the experiment or a function that generates parameter values. The function has to be format (oldvalue) -> (newvalue), Third: OPTIONAL: a function that maps (old_parameter_instance; (par_range[1],new_parameter_value)) -> new_parameter_instance. Default is 'reconstruct' from @with_kw/Parameters.jl is used) For examples see [`Basic Usage`](@ref).
* `eval_ode_func`: Evaluation function for the MonteCarloProblem with the signature `(sol,i)->(results, repeat)`. There are many premade functions for this purpose in this library, most of them called `eval_ode_run`, see also [`eval_ode_run`](@ref)
* `tail_frac`: Only after this time (relative to the total integration time) the solutions are evaluated

## Non-randomized initial conditions

    DEMCBBProblem(p::DiffEqBase.DEProblem, ic_ranges::Array{<:AbstractArray,1}, pars::DEParameters, par_range_tuple::ParameterVar, eval_ode_func::Function, tail_frac::Number)

Setup a DEMCBBProblem with initial conditions (and parameters) from predefined arrays or ranges.
All arguments are identical to the other constructor except for:
* `ic_ranges`: A range/array or array of ranges/arrays with initial conditions for each trial. If only one range/array is provided its used for all IC dims.
* Note that there is _no_ `N_ic` argument in constrast to the other constructor

## Direct constructor

It is also possible to initialize the type directly with its fields with

    DEMCBBProblem(p::MonteCarloProblem, N_mc::Int64, rel_transient_time::Float64, ic::AbstractArray, par::AbstractArray, par_range_tuple::ParameterVar)
"""
struct DEMCBBProblem <: MCBBProblem
    p::MonteCarloProblem
    N_mc::Int64
    rel_transient_time::Float64
    ic::AbstractArray
    par::AbstractArray
    par_var::ParameterVar

    function DEMCBBProblem(p::DiffEqBase.DEProblem, ic_gens::Array{<:Function,1}, N_ic::Int, pars::DEParameters, par_range_tuple::ParameterVar, eval_ode_func::Function, tail_frac::Number)
        (ic_coupling_problem, ic, par, N_mc) = setup_ic_par_mc_problem(p, ic_gens, N_ic, pars, par_range_tuple)
        mcp = MonteCarloProblem(p, prob_func=ic_coupling_problem, output_func=eval_ode_func)
        new(mcp, N_mc, tail_frac, ic, par, par_range_tuple)
    end

    function DEMCBBProblem(p::DiffEqBase.DEProblem, ic_ranges::Array{<:AbstractArray,1}, pars::DEParameters, par_range_tuple::ParameterVar, eval_ode_func::Function, tail_frac::Number)
        (ic_coupling_problem, ic, par, N_mc) = setup_ic_par_mc_problem(p, ic_ranges, pars, par_range_tuple)
        mcp = MonteCarloProblem(p, prob_func=ic_coupling_problem, output_func=eval_ode_func)
        new(mcp, N_mc, tail_frac, ic, par, par_range_tuple)
    end

    # Direct Constructor
    DEMCBBProblem(p::MonteCarloProblem, N_mc::Int64, rel_transient_time::Float64, ic::AbstractArray, par::AbstractArray, par_range_tuple::ParameterVar) = new(p, N_mc, rel_transient_time, ic, par, par_range_tuple)
end
DEMCBBProblem(p::DiffEqBase.DEProblem, ic_gens::Function, N_ic::Int, pars::DEParameters, par_range_tuple::ParameterVar, eval_ode_func::Function, tail_frac::Number) = DEMCBBProblem(p, [ic_gens], N_ic, pars, par_range_tuple, eval_ode_func, tail_frac)
DEMCBBProblem(p::DiffEqBase.DEProblem, ic_gens::Union{Array{<:Function,1},Function}, N_ic::Int, pars::DEParameters, par_range_tuple::ParameterVar, eval_ode_func::Function) = DEMCBBProblem(p,ic_gens,N_ic,pars,par_range_tuple,eval_ode_func, 0.9)

# automaticlly convert appropiate tuples to ParameterVar
DEMCBBProblem(p::DiffEqBase.DEProblem, ic_gens::Array{<:Function,1}, N_ic::Int, pars::DEParameters, par_range_tuple::Union{Tuple{Symbol,Union{AbstractArray,Function},<:Function},Tuple{Symbol,Union{AbstractArray,Function}}}, eval_ode_func::Function, tail_frac::Number) = DEMCBBProblem(p,ic_gens, N_ic, pars, OneDimParameterVar(par_range_tuple...), eval_ode_func, tail_frac)
DEMCBBProblem(p::DiffEqBase.DEProblem, ic_ranges::Array{<:AbstractArray,1}, pars::DEParameters, par_range_tuple::Union{Tuple{Symbol,Union{AbstractArray,Function},<:Function},Tuple{Symbol,Union{AbstractArray,Function}}}, eval_ode_func::Function, tail_frac::Number) = DEMCBBProblem(p, ic_ranges, pars, OneDimParameterVar(par_range_tuple...), eval_ode_func, tail_frac)
DEMCBBProblem(p::DiffEqBase.DEProblem, ic_gens::Function, N_ic::Int, pars::DEParameters, par_range_tuple::Union{Tuple{Symbol,Union{AbstractArray,Function},<:Function},Tuple{Symbol,Union{AbstractArray,Function}}}, eval_ode_func::Function, tail_frac::Number) = DEMCBBProblem(p, ic_gens, N_ic, pars, OneDimParameterVar(par_range_tuple...), eval_ode_func, tail_frac)
DEMCBBProblem(p::DiffEqBase.DEProblem, ic_gens::Union{Array{<:Function,1},Function}, N_ic::Int, pars::DEParameters, par_range_tuple::Union{Tuple{Symbol,Union{AbstractArray,Function} ,<:Function},Tuple{Symbol,Union{AbstractArray,Function}}}, eval_ode_func::Function) = DEMCBBProblem(p,ic_gens,N_ic,pars,OneDimParameterVar(par_range_tuple...),eval_ode_func)

"""
    parameter(p::DEMCBBProblem, i::Int=1; complex_returns_abs=true)

Utility function that returns the parameters of each trial of of a problem. In case multiple parameters are varied simultaneously it returns the `i`-th parameter. In case the initial conditions or parameters are complex valued the function returns the absolute value of the parameters if `complex_returns_abs==true` and the original complex number if `complex_returns_abs==false`.
"""
function parameter(p::MCBBProblem, i::Int; complex_returns_abs=true)
    if (eltype(p.par)<:Complex) && complex_returns_abs
        return abs.(p.par[:,i])
    else
        return p.par[:,i]
    end
end

function parameter(p::MCBBProblem; complex_returns_abs=true)
    if length(p.par_var)>1
        parvals = p.par
    else
        parvals = p.par[:,1]
    end

    if (eltype(p.par)<:Complex) && complex_returns_abs
        return abs.(parvals)
    else
        return parvals
    end
end

"""
    MCBBSol <: myMCSol

Abstract type of [`DEMCBBSol`](@ref) (using DifferentialEquations.jl as a backend) and [`CustomMCBBSolution`](@ref) (using customized `solve` and problem functions).
"""
abstract type MCBBSol <: myMCSol end

"""
    DEMCBBSol

Type that stores the solutions of a DEMCBBProblem. Is returned by the corresponding `solve` routine.

Its fields are:
* `sol`: MonteCarloSolution (see DifferentialEquations)
* `N_mc`: number of solutions saved / Monte Carlo trials runs
* `N_t`: number of time steps for each solutions
* `N_dim`: sytem dimension
* `N_meas`: number of measures used, ``N_{meas} = N_{meas_{dim}} + N_{meas_{global}}
* `N_meas_dim`: number of measures that are evalauted for every dimension seperatly
* `N_meas_global`: number of measures that are evalauted globally
* `solve_command`: A function that solves one individual run/trial with the same settings as where used to

Note, in case `N_dim==1` => `N_meas_global == 0` and `N_meas_dim == N_meas`
"""
struct DEMCBBSol <: MCBBSol
    sol::MonteCarloSolution
    N_mc::Int
    N_t::Int
    N_dim::Int
    N_meas::Int
    N_meas_dim::Int
    N_meas_global::Int
    N_meas_matrix::Int
    solve_command::Function
end

"""
    sort(sol::DEMCBBSol, prob::DEMCBBProblem, i::Int=1)

Returns a copy of the `sol` and `prob` sorted by the values of the `i`-th parameter.

(In case the parameter is complex valued, it sorts by the absolute value of the parameter)
"""
function sort(sol::MCBBSol, prob::MCBBProblem, i::Int=1)
    sol_copy = deepcopy(sol)
    prob_copy = deepcopy(prob)
    sort!(sol_copy, prob, i)
    sort!(prob_copy, i)
    (sol_copy, prob_copy)
end

"""
    sort(sol::DEMCBBSol, prob::DEMCBBProblem, i::Int=1)

Sorts `sol` and `prob` inplace by the values of the `i`-th parameter.

(In case the parameter is complex valued, it sorts by the absolute value of the parameter)
"""
function sort!(sol::MCBBSol, prob::MCBBProblem, i::Int=1)
    p = parameter(prob, i)
    if eltype(p) <: Complex
        sortind = sortperm(abs.(p))
    else
        sortind = sortperm(p)
    end
    prob.ic[:,:] = prob.ic[sortind,:]
    prob.par[:,:] = prob.par[sortind,:]
    # this sort of indexing with doesnt work on some versions, couldnt figure out why exactly, so I have to code the sorting by hand
    # sol.sol[:] = sol.sol[sortind]

    sol_copy = deepcopy(sol.sol)
    for (i,i_perm) in enumerate(sortind)
        sol.sol[i] = sol_copy[i_perm]
    end
end

"""
    sort(prob::DEMCBBProblem, i::Int=1)

Returns a copy of `prob` sorted by the values of the `i`-th paramater.

(In case the parameter is complex valued, it sorts by the absolute value of the parameter)
"""
function sort(prob::MCBBProblem, i::Int=1)
    prob_copy = deepcopy(prob)
    sort!(prob_copy, i)
    prob_copy
end

"""
    sort(prob::DEMCBBProblem, i::Int=1)

Sorts `prob` inplace by the values of the`i`-th paramater.

(In case the parameter is complex valued, it sorts by the absolute value of the parameter)
"""
function sort!(prob::MCBBProblem, i::Int=1)
    p = parameter(prob, i)
    if eltype(p) <: Complex
        sortind = sortperm(abs.(p))
    else
        sortind = sortperm(p)
    end
    prob.ic[:,:] = prob.ic[sortind,:]
    prob.par[:,:] = prob.par[sortind,:]
end

"""
    show_results(sol::DEMCBBSol, prob::DEMCBBProblem, min_par::Number, max_par::Number, i::Int=1, sorted::Bool=false)

Returns the solutions for which the `i`-th parameter exhibits values between `min_par` and `max_par`.
If `sorted==true` it assumes that `sol` and `prob` are already sorted by this parameter.
"""
function show_results(sol::DEMCBBSol, prob::DEMCBBProblem, min_par::Number, max_par::Number, i::Int=1, sorted::Bool=false)
    if sorted==false
        (ssol, sprob) = sort(sol, prob, i)
    else
        ssol = sol
        sprob = prob
    end

    p = parameter(sprob, i)
    ssol.sol.u[(p .> min_par) .& (p .< max_par)]
end

"""
    get_measure(sol::MCBBSol, k::Int)

Return the results for the `k`-th measure as an array.
"""
function get_measure(sol::MCBBSol, k::Int)
    if k <=  sol.N_meas_dim
        arr = zeros(eltype(sol.sol[1][k]),(sol.N_mc,sol.N_dim))
        for i=1:sol.N_mc
            arr[i,:] = sol.sol[i][k]
        end
    elseif k <= sol.N_meas_dim + sol.N_meas_matrix
        arr = zeros(eltype(sol.sol[1][k]),(sol.N_mc, size(sol.sol[1][k])...))
        for i=1:sol.N_mc
            arr[i,:,:] = sol.sol[i][k]
        end
    else
        arr = zeros(typeof(sol.sol[1][k]),sol.N_mc)
        for i=1:sol.N_mc
            arr[i] = sol.sol[i][k]
        end
    end
    arr
end

"""
    normalize(sol::DEMCBBSol, k::AbstractArray)

Returns a copy of `sol` with the solutions normalized to be in range [0,1]. It is possible to only select that some of the measures are normalized by providing an array with the indices of the measures that should be normalized, e.g. `[1,2]` for measure 1 and measure 2 to be normalized.

    normalize(sol::DEMCBBSol)

If no extra array is provided, all measures are normalized.
"""
function normalize(sol::DEMCBBSol, k::AbstractArray)
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
        if i_meas < (sol.N_meas_dim + sol.N_meas_matrix)
            for i_mc=1:sol.N_mc
                new_mc_sol.u[i_mc][i_meas][:] = (sol.sol.u[i_mc][i_meas][:] .- min_meas[i_meas]) ./ meas_ranges[i_meas]
            end
        else
            @warn "Global measures are not normlized."
        end
    end

    sol_new = DEMCBBSol(new_mc_sol, sol.N_mc, sol.N_t, sol.N_dim, get_measure_dimensions(sol)..., sol.solve_command)
end
normalize(sol::DEMCBBSol) = normalize(sol, 1:sol.N_meas)

"""
    setup_ic_par_mc_problem

Methods that are usually called automaticly while constructing a `DEMCBBProblem`. These methods setup the initial conditions - parameter matrix and the problem function for the `MonteCarloProblem`.


# Initial Condtions set with Arrays or Ranges

    setup_ic_par_mc_problem(prob::DiffEqBase.DEProblem, ic_ranges::Array{T,1}, parameters::DEParameters, var_par::ParameterVar) where T <: AbstractArray

* `prob`: A Problem from DifferentialEquations, currently supported are `DiscreteProblem`, `ODEProblem`, `SDEProblem`, the base problem one is interested in.
* `ic_ranges`: A range/array or array of ranges/arrays with initial conditions for each trial. If only one range/array is provided its used for all IC dims.
* `parameters`: parameter struct of the underlying system
* `var_par`:  `ParameterVar`, information about how the parameters are varied, see [`ParameterVar`](@ref). Its also possible to hand over an appropiate tuple that will be automaticly converted to a `ParameterVar` type. For examples see [`Basic Usage`](@ref).

# Initial Condtions generated by Functions

    setup_ic_par_mc_problem(prob::DiffEqBase.DEProblem, ic_gens::Array{<:Function,1}, N_ic::Int, parameters::DEParameters, var_par::ParameterVar)

* `prob`: A Problem from DifferentialEquations, currently supported are `DiscreteProblem`, `ODEProblem`, `SDEProblem`, the base problem one is interested in.
* `ic_gens`: A function or an array of functions that generate the initial conditions for each trial.  Function signature is `()->new_value::Number` or `(i_run)->new_value::Number`. If only one function is provided its used for all IC dims, if ``M<N_{dim}`` functions with ``N_{dim}=k\\cdot M`` are provided these functions are repeated ``k`` times (useful e.g. for coupled chaotic oscillators).
* `parameters`: parameter struct of the underlying system
* `var_par`:  `ParameterVar`, information about how the parameters are varied, see [`ParameterVar`](@ref).
"""
function setup_ic_par_mc_problem(prob, ic_ranges::Array{T,1}, parameters::DEParameters, var_par::ParameterVar) where T <: AbstractArray
    N_dim_ic = length(prob.u0)
    if N_dim_ic != length(ic_ranges)
        error("Number of IC arrays/ranges doesnt match system dimension")
    end
    N_dim = N_dim_ic + length(var_par)

    (ic, par, N_mc) = _ic_par_matrix(N_dim_ic, N_dim, ic_ranges, var_par)

    ic_par_problem = define_new_problem(prob, ic, par, parameters, N_dim_ic, ic_ranges, var_par)
    (ic_par_problem, ic, par, N_mc)
end

function setup_ic_par_mc_problem(prob, ic_gens::Array{<:Function,1}, N_ic::Int, parameters::DEParameters, var_par::ParameterVar)
    N_dim_ic = length(prob.u0)
    N_dim = N_dim_ic + length(var_par)

    ic_gens = verify_func.(ic_gens)
    (ic, par, N_mc) = _ic_par_matrix(N_dim_ic, N_dim, N_ic, ic_gens, var_par)
    ic_par_problem = define_new_problem(prob, ic, par, parameters, N_dim_ic, ic_gens, var_par)
    (ic_par_problem, ic, par, N_mc)
end
setup_ic_par_mc_problem(prob, ic_gens::Function, N_ic::Int, parameters::DEParameters, var_par::ParameterVar) = setup_ic_par_mc_problem(prob, [ic_gens], N_ic, parameters, var_par)

"""
    define_new_problem(prob, ic::Abstract, par::AbstractArray, parameters::DEParameters, N_dim_ic::Int, ic_gens::AbstractArray, var_par::ParameterVar)

Returns the functions that returns new DifferentialEquations problems needed for `MonteCarloProblem`.

* `prob`: A Problem from DifferentialEquations, currently supported are `DiscreteProblem`, `ODEProblem`, `SDEProblem`, `DDEProblem` the base problem one is interested in.
* `ic`: (``N_{mc} \\times N_{dim_{ic}}``)-Matrix containing initial conditions for each run.
* `par`: (``N_{mc} \\times N_{par}``)-Matrix containing parameter values for each run.
* `N_dim_ic`: system dimension
* `ic_gens`: Array of functions or arrays/ranges that contain/generate the ICs.
* `var_par`:  `ParameterVar`, information about how the parameters are varied, see [`ParameterVar`](@ref).
"""
function define_new_problem(prob::ODEProblem, ic::AbstractArray, par::AbstractArray, parameters::DEParameters, N_dim_ic::Int, ic_gens::AbstractArray, var_par::ParameterVar)
    function new_problem(prob, i, repeat)
        _repeat_check(repeat, i, ic, ic_gens, N_dim_ic)
        ODEProblem(prob.f, ic[i,:], prob.tspan,  var_par.new_par(parameters; _new_val_dict(var_par, par, N_dim_ic, i)...), callback=prob.callback)
    end
    new_problem
end

function define_new_problem(prob::DiscreteProblem, ic::AbstractArray, par::AbstractArray, parameters::DEParameters, N_dim_ic::Int, ic_gens::AbstractArray, var_par::ParameterVar)
    function new_problem(prob, i, repeat)
        _repeat_check(repeat, i, ic, ic_gens, N_dim_ic)
        DiscreteProblem(prob.f, ic[i,:], prob.tspan,  var_par.new_par(parameters; _new_val_dict(var_par, par, N_dim_ic, i)...), callback=prob.callback)
    end
    new_problem
end

function define_new_problem(prob::SDEProblem, ic::AbstractArray, par::AbstractArray, parameters::DEParameters, N_dim_ic::Int, ic_gens::AbstractArray, var_par::ParameterVar)
    function new_problem(prob, i, repeat)
        _repeat_check(repeat, i, ic, ic_gens, N_dim_ic)
        SDEProblem(prob.f, prob.g, ic[i,:], prob.tspan,  var_par.new_par(parameters; _new_val_dict(var_par, par, N_dim_ic, i)...), callback=prob.callback)
    end
    new_problem
end

function define_new_problem(prob::DDEProblem, ic::AbstractArray, par::AbstractArray, parameters::DEParameters, N_dim_ic::Int, ic_gens::AbstractArray, var_par::ParameterVar)
    function new_problem(prob, i, repeat)
        _repeat_check(repeat, i, ic, ic_gens, N_dim_ic)
        DDEProblem(prob.f, ic[i,:], prob.h, prob.tspan,  var_par.new_par(parameters; _new_val_dict(var_par, par, N_dim_ic, i)...); constant_lags=prob.constant_lags, dependent_lags=prob.dependent_lags, callback=prob.callback)
    end
    new_problem
end

"""
    _new_val_dict(var_par::ParameterVar, par::AbstractArray, N_dim_ic::Int, i::Int)

Returns a dictionary with the names of the parameter fields to be varied and their for the new values for the `i`-th run. Used to get a new parameter instance with this dictionary as the keyword argument.

* `var_par`:  `ParameterVar`, information about how the parameters are varied, see [`ParameterVar`](@ref).
* `par`: (``N_{mc} \\times N_{par}``)-Matrix containing parameter values for each run.
* `N_dim_ic`: system dimension
* `i`: number of run/trial
"""
function _new_val_dict(var_par::MultiDimParameterVar, par::AbstractArray, N_dim_ic::Int, i::Int)
    par_arr = []
    N_dim_par = length(var_par)
    for i_par=1:N_dim_par
        push!(par_arr, (var_par[i_par].name, par[i,i_par]))
    end
    Dict(par_arr)
end
function _new_val_dict(var_par::OneDimParameterVar, par::AbstractArray, N_dim_ic::Int, i::Int)
    Dict(var_par.name => par[i,1])
end

"""
    verify_func(func::Function)

Quick and dirty way to convert function with signature `()->result` to `(i)->result`. Note that this actually converts all function with a signature other than func(i::Number) to (i::Number)->func(). This will cause errors if it is used on function other than ()->result.
"""
function verify_func(func::Function)
    try
        func(1)
    catch excp
        if isa(excp, MethodError)
            return ((i)->func())
        end
    end
    return func
end


"""
    _repeat_check(repeat, ic::AbstractArray, ic_gens)

Checks if the problem has to be repeated, if so, it generates new ICs.

* `repeat`: Boolean, returns from `MonteCarloProblem`
* `ic`: (``N_{mc} \\times N_{dim_{ic}}``)-Matrix containing initial conditions for each run.
* `ic_gens`: Array of functions or arrays/ranges that contain/generate the ICs.
"""
function _repeat_check(repeat, i::Int, ic::AbstractArray, ic_gens::Array{<:Function,1}, N_dim_ic::Int; verbose=true)
    if repeat > 1
        if repeat > 10
            println("------------------")
            println("Error with IC:")
            println(ic[i,:])
            println("------------------")
            error("More than 10 Repeats of a Problem in the Monte Carlo Run, there might me something wrong here!")
        else
            if verbose
                print("run ")
                print(i)
                print(" repeated with IC config: ")
                println(ic[i,:])
            end
            ic[i,:] = _new_ics(i,N_dim_ic,ic_gens)
        end
    end
end
function _repeat_check(repeat, i::Int, ic::AbstractArray, ic_gens::Array{T,1}, N_dim_ic::Int) where T<:AbstractArray
    if repeat > 1
        error("Problem has to be repeated, but ICs are non-random, it would result in the same solution!")
    end
end

"""
    _ic_par_matrix

Methods that compute `ic` and `par`, (``N_{mc} \\times (N_{dim_{ic}} + N_{par})``)-Matrices containing initial conditions and parameter values for each run. Returns `(ic, par, N_mc)`.

# Initial Conditions and Parameters from arrays/ranges

    _ic_par_matrix(N_dim_ic::Int, N_dim::Int, ic_ranges::Array{T,1}, var_par::MultiDimParameterVarArray) where T <: AbstractArray

* `N_dim_ic`: system dimension
* `N_dim`: system dimension + number of parameters that are varied
* `ic_ranges`: A range/array or array of ranges/arrays with initial conditions for each trial. If only one range/array is provided its used for all IC dims.
* `var_par`: `ParameterVar`, either `MultiDimParameterVarArray` or `ParameterVarArray`

# Initial Conditions and Parameters from functions (randomly) generated

    _ic_par_matrix(N_dim_ic::Int, N_dim::Int, N_ic::Int, ic_gens::Array{<:Function,1},  var_par::MultiDimParameterVarFunc)

* `N_dim_ic`: system dimension
* `N_dim`: system dimension + number of parameters that are varied
* `ic_gens`: A function or an array of functions that generate the initial conditions for each trial.  Function signature is `()->new_value::Number`. If only one function is provided its used for all IC dims, if ``M<N_{dim}`` functions with ``N_{dim}=k\\cdot M`` are provided these functions are repeated ``k`` times (useful e.g. for coupled chaotic oscillators).
* `var_par`: `ParameterVar`, either `MultiDimParameterVarFunc` or `ParameterVarFunc`

# Initial Conditions random, Parameters from arrays/ranges

    _ic_par_matrix(N_dim_ic::Int, N_dim::Int, N_ic::Int, ic_gens::Array{T,1},  var_par::MultiDimParameterVarArray) where T<:Function

If the parameters are varied along arrays/ranges and the initial conditions are generated (randomly) by functions, the initial conditions are equal for each parameter value.

* `N_dim_ic`: system dimension
* `N_dim`: system dimension + number of parameters that are varied
* `ic_gens`: A function or an array of functions that generate the initial conditions for each trial.  Function signature is `()->new_value::Number`. If only one function is provided its used for all IC dims, if ``M<N_{dim}`` functions with ``N_{dim}=k\\cdot M`` are provided these functions are repeated ``k`` times (useful e.g. for coupled chaotic oscillators).
* `var_par`: `ParameterVar`, either `MultiDimParameterVarArray` or `ParameterVarArray`
"""
_ic_par_matrix(N_dim_ic::Int, N_dim::Int, ic_ranges::Array{<:AbstractArray,1}, var_par::ParameterVarArray) = _ic_par_matrix(N_dim_ic, N_dim, ic_ranges, MultiDimParameterVar(var_par))
function _ic_par_matrix(N_dim_ic::Int, N_dim::Int, ic_ranges::Array{T,1}, var_par::MultiDimParameterVarArray) where T <: AbstractArray
    N_ic_pars = zeros(Int, N_dim)
    for (i_range, ic_range) in enumerate(ic_ranges)
        N_ic_pars[i_range] = length(collect(ic_range))
    end
    for i_par=1:length(var_par)
        N_ic_pars[N_dim_ic+i_par] = var_par[i_par].N
    end

    if prod(float(N_ic_pars)) > 1e10
        @warn "More than 1e10 initial cond. Are you sure what you are doing? Overflows might occur."
    end

    N_mc = prod(N_ic_pars)
    if N_mc==0
        error("Zero inital conditions. Either at least one of the ranges has length 0 or an overflow occured")
    end

    N_ic_pars = tuple(N_ic_pars...) # need this as tuple for CartesianIndices

    ic = zeros(eltype(ic_ranges[1]),(N_mc, N_dim_ic))
    par = zeros(typeof(var_par[1].new_val(1)),(N_mc, length(var_par)))
    for (i_mc, i_ci) in enumerate(CartesianIndices(N_ic_pars))
         for i_dim=1:N_dim_ic
             ic[i_mc, i_dim] = ic_ranges[i_dim][i_ci[i_dim]]
         end
         for i_par=1:length(var_par)
             par[i_mc, i_par] = var_par[i_par].new_val(i_ci[N_dim_ic+i_par])
         end
    end
    (ic, par, N_mc)
end

# helper function for setup_ic_par_mc_problem()

# uses random generators for IC and Parameter
_ic_par_matrix(N_dim_ic::Int, N_dim::Int, N_ic::Int, ic_gens::Array{<:Function,1},  var_par::ParameterVarFunc) = _ic_par_matrix(N_dim_ic, N_dim, N_ic, ic_gens, MultiDimParameterVar(var_par))
function _ic_par_matrix(N_dim_ic::Int, N_dim::Int, N_ic::Int, ic_gens::Array{<:Function,1},  var_par::MultiDimParameterVarFunc)
    N_gens = length(ic_gens) # without the parameter geneartor
    if N_dim_ic % (N_gens) != 0
        err("Number of initial cond. genators and Number of initial cond. doesn't fit together")
    end
    N_gen_steps = Int(N_dim_ic / N_gens)

    # check if ic gens returns a single value or all ics at the same time as an array
    if (typeof(ic_gens[1](1))<:AbstractArray) & (N_gens == 1)
        ic = zeros(eltype(ic_gens[1](1)),(N_ic, N_dim_ic))
        par = zeros(typeof(var_par[1].new_val(1)),(N_ic, length(var_par)))

        for i_ic=1:N_ic
            ic[i_ic,:] = ic_gens[1](i_ic)
            for i_par=1:length(var_par)
                par[i_ic, i_par] = var_par[i_par].new_val(i_ic)
            end
        end
    else
        # ics per dim
        ic = zeros(typeof(ic_gens[1](1)),(N_ic, N_dim_ic))
        par = zeros(typeof(var_par[1].new_val(1)),(N_ic, length(var_par)))

        for i_ic=1:N_ic
            for i_gen_steps=1:N_gen_steps
                for i_gen=1:N_gens
                    ic[i_ic, N_gens*i_gen_steps - (N_gens - i_gen)] = ic_gens[i_gen](i_ic)
                end
            end
            for i_par=1:length(var_par)
                par[i_ic, i_par] = var_par[i_par].new_val(i_ic)
            end
        end
    end
    (ic, par, N_ic)
end

# helper function for setup_ic_par_mc_problem()
# uses (random) generator functions for the initial cond. and ranges for Parameter
# sets it up so that each parameter value has identical ICs
_ic_par_matrix(N_dim_ic::Int, N_dim::Int, N_ic::Int, ic_gens::Array{<:Function,1},  var_par::ParameterVarArray) = _ic_par_matrix(N_dim_ic, N_dim, N_ic, ic_gens, MultiDimParameterVar(var_par))
function _ic_par_matrix(N_dim_ic::Int, N_dim::Int, N_ic::Int, ic_gens::Array{T,1},  var_par::MultiDimParameterVarArray) where T<:Function
    if length(var_par)>1
        error("Random IC and Parameter with Ranges is so far only supported for N_par=1")
    end
    N_ic_pars = (N_ic, var_par[1].N)
    N_mc = prod(N_ic_pars)
    N_gens = length(ic_gens)
    if N_dim_ic % N_gens != 0
        err("Number of initial cond. genators and Number of initial cond. doesn't fit together")
    end
    N_gen_steps = Int(N_dim_ic / N_gens)

    # check if ic gens returns a single value or all ics at the same time as an array
    if (typeof(ic_gens[1](1))<:AbstractArray) & (N_gens == 1)
        ics = zeros(eltype(ic_gens[1](1)),(N_ic, N_dim_ic))
        for i_ic=1:N_ic
            ics[i_ic, 1:N_dim_ic] = ic_gens[1](i_ic)
        end
        ic = zeros(eltype(ic_gens[1](1)),(N_mc, N_dim_ic))
        par = zeros(typeof(var_par[1].new_val(1)),(N_mc, length(var_par)))
    else
        # ics per dim
        ics = zeros(typeof(ic_gens[1](1)),(N_ic, N_dim_ic))
        for i_ic=1:N_ic
            for i_gen_steps=1:N_gen_steps
                for i_gen=1:N_gens
                    ics[i_ic, N_gens*i_gen_steps - (N_gens - i_gen)] = ic_gens[i_gen](i_ic)
                end
            end
        end
        ic = zeros(typeof(ic_gens[1](1)),(N_mc, N_dim_ic))
        par = zeros(typeof(var_par[1].new_val(1)),(N_mc, length(var_par)))
    end

    for (i_mc, i_ci) in enumerate(CartesianIndices(N_ic_pars))
        ic[i_mc,:] = ics[i_ci[1],:]
        for i_par=1:length(var_par)
            par[i_mc, i_par] = var_par[i_par].new_val(i_ci[2])
        end
    end
    (ic, par, N_mc)
end

"""
    _new_ics(i::Int, N_dim_ic::Int, ic_gens::Array{T,1}) where T<:Function

Helper functions, calculate new ICs in case the MonteCarlo trial needs to be repeated.
"""
function _new_ics(i::Int, N_dim_ic::Int, ic_gens::Array{T,1}) where T<:Function
    N_gens = length(ic_gens)
    N_gen_steps = Int(N_dim_ic / N_gens)

    # check if ic gens return an array for all ics at one or just a single value
    if (typeof(ic_gens[1](1))<:AbstractArray) & (N_gens == 1)
        ics = ic_gens[1](i)
    else
        ics = zeros(typeof(ic_gens[1](i)), N_dim_ic)
        for i_gen_steps=1:N_gen_steps
            for i_gen=1:N_gens
                ics[N_gens*i_gen_steps - (N_gens - i_gen)] = ic_gens[i_gen](i)
            end
        end
    end
    ics
end

"""
    solve(prob::DEMCBBProblem, alg=nothing, N_t=400::Int, parallel_type=:parfor; flag_check_inf_nan=true, custom_solve::Union{Function,Nothing}=nothing, kwargs...)

Custom solve for the `DEMCBBProblem`. Solves the `MonteCarloProblem`, but saves and evaluates only after transient at a constant step size, the results are sorted by parameter value.

* `prob`: `MonteCarloProblem` of type defined in this library
* `alg`: Algorithm to use, same as for `solve()` from DifferentialEquations.jl
* `N_t`: Number of timesteps to be saved
* `parallel_type`: Which form of parallelism should be used? Same as for `MonteCarloProblem` from DifferentialEquations.jl
* `flag_check_inf_nan`: Does a check if any of the results are `NaN` or `inf`
* `custom_solve`:: Function/Nothing, custom solve function
"""
function solve(prob::DEMCBBProblem, alg=nothing, N_t=400::Int, parallel_type=:parfor; flag_check_inf_nan=true, custom_solve::Union{Function,Nothing}=nothing, sort_results=true, kwargs...)
    t_save = collect(tsave_array(prob.p.prob, N_t, prob.rel_transient_time))

    if custom_solve!=nothing
        sol = custom_solve(prob, t_save)
        solve_i_command = nothing
    elseif alg!=nothing
        sol = solve(prob.p, alg, num_monte=prob.N_mc, dense=false, save_everystep=false, saveat=t_save, savestart=false, parallel_type=parallel_type; kwargs...)
        solve_i_command = (prob_i) -> solve(prob_i, alg=alg, dense=false, save_everystep=false, saveat_t_save, savestart=false; kwargs...)
    else
        sol = solve(prob.p, num_monte=prob.N_mc, dense=false, save_everystep=false, saveat=t_save, savestart=false, parallel_type=parallel_type; kwargs...)
        solve_i_command = (prob_i) -> solve(prob_i, dense=false, save_everystep=false, saveat=t_save, savestart=false; kwargs...)
    end


    ___, N_dim = size(prob.ic)

    mysol = DEMCBBSol(sol, prob.N_mc, N_t, length(prob.p.prob.u0), get_measure_dimensions(sol, N_dim)..., solve_i_command)

    if flag_check_inf_nan
        inf_nan = check_inf_nan(mysol)
        if (length(inf_nan["Inf"])>0) | (length(inf_nan["NaN"])>0)
            @warn "Some elements of the solution are Inf or NaN, check the solution with check_inf_nan again!"
        end
    end
    if sort_results
        sort!(mysol, prob)
    end
    mysol
end
solve_euler_inf(prob::DEMCBBProblem, t_save::AbstractArray; dt=0.1) = solve(prob.p, alg=Euler(), dt=dt, num_monte=prob.N_mc, parallel_type=:parfor, dense=false, saveat=t_save, tstops=t_save, savestart=false, save_everystep=false)

"""
    get_measure_dimensions(sol)

Returns the number of measures of `sol` as a tuple `(N_meas_total, N_meas_dim, N_meas_global)`
"""
function get_measure_dimensions(sol, N_dim)
    sol1 = sol[1] # use the first solution for the dimension determination

    N_meas_dim = 0
    N_meas_global = 0
    N_meas_matrix = 0

    for i=1:length(sol1)
        if typeof(sol1[i]) <: AbstractArray
            if length(sol1[i])==N_dim
                N_meas_dim += 1
            else
                N_meas_matrix += 1
            end
        else
            N_meas_global += 1
        end
    end
    (N_meas_dim + N_meas_global + N_meas_matrix, N_meas_dim, N_meas_global, N_meas_matrix)
end
get_measure_dimensions(sol::MCBBSol) = get_measure_dimensions(sol.sol, sol.N_dim)

"""
     tsave_array(prob, N_t::Int, rel_transient_time::Float64=0.7)

Given a tspan to be used in a DEProblem, returns the array/iterator of time points to be saved (saveat argument of solve())

* `prob`: DifferentialEquations problem
* `N_t` : number of time points to be saved
* `rel_transient_time`: Only after this time (relative to the total integration time) the solutions are evaluated
"""
function tsave_array(prob::Union{ODEProblem,SDEProblem,DDEProblem}, N_t::Int, rel_transient_time::Float64=0.7)
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

"""
    Base.zero(a::Type{Char}) = '0'

Helper function needed for initializing zero char arrays in case we work for example with SIR models.
"""
Base.zero(a::Type{Char}) = '0'
