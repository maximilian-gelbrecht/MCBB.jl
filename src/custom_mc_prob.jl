# This file contains functions concerning the setup of a custom problem type for systems that can't be solved with DifferentialEquations.jl as a backend
using Distributed
using DifferentialEquations
import LinearAlgebra.normalize

import DifferentialEquations.solve # this needs to be directly importet in order to extend it with our own solve() for our own problem struct

"""
    CustomProblem

Structure that emulates some of the fields/behaviour of `DEProblem` subtypes from `DifferentialEquations`.
The results are supposed to be a (N_dim x N_t)-array.

Fields:
* `f`: Problem function, signature: `(u0, p, tspan) -> results`
* `u0::AbstractArray`: Initial conditions
* `p`: Parameters
* `tspan`: Time Span

"""
struct CustomProblem
    f
    u0::AbstractArray
    tspan
    p
end

"""
    solve(prob::CustomProblem)

Solves/runs the system/experiment with the initial conditions and parameters that were set during construction of `prob`.
"""
solve(prob::CustomProblem) = CustomSolution(prob.f(prob.u0, prob.p, prob.tspan), prob)

"""
    CustomSolution

Solution type of `CustomProblem`. Contains the solution as an abstract array and the problem. Can be index like an array.

Fields:
* `sol::AbstractArray`: Solution as an Array (1d (`N_t`-long Array) or 2d (`N_dim` x `N_t`s)-sized Array)
* `prob::CustomProblem`: Problem that was solved to get the solution

"""
struct CustomSolution
    sol::AbstractArray
    prob::CustomProblem

    function CustomSolution(sol::AbstractArray, prob::CustomProblem)
        if ndims(sol)==1
            new_sol = reshape(sol, (1,length(sol)))
        elseif ndims(sol)>2
            error("sol has more than 2-d. Thats wrong! It should be either a N_t-long 1d array or a (N_dim x N_t) array")
        else
            new_sol = sol
        end
        new(new_sol, prob)
    end
end

Base.getindex(sol::CustomSolution, ind...) = getindex(sol.sol, ind...)
Base.setindex!(sol::CustomSolution, val, inds...) = setindex!(sol.sol, val, inds...)
Base.length(sol::CustomSolution) = length(sol.sol)
Base.firstindex(sol::CustomSolution) = 1
Base.lastindex(sol::CustomSolution) = length(sol.sol)
Base.lastindex(sol::CustomSolution, i::Int) = lastindex(sol.sol, i)
Base.size(sol::CustomSolution) = size(sol.sol)
Base.Matrix(sol::CustomSolution) = sol.sol

"""
    CustomMonteCarloProblem

Structure similar to `DifferentialEquations`'s `MonteCarloProblem` but for problems that can not be solved with `DifferentialEquations`. The fields emululate those of `MonteCarloProblem`:

* `prob::CustomProblem`: The base problem that should be solved
* `prob_func::Function`: same as for `MonteCarloProblem`: A function `(prob, i, repeat) -> new_problem` that returns the i-th problem that should be solved
* `eval_func::Function`: same as for `MonteCarloProblem`: A function `(sol, i) -> (results, repeat)` that evaluated the i-th solution
"""
struct CustomMonteCarloProblem
    prob::CustomProblem
    prob_func::Function
    eval_func::Function
end

"""
    solve(prob::CustomMonteCarloProblem; num_monte::Int=100)

Solves the `CustomMonteCarloProblem` `num_monte`-times.

TO-DO: parallelization!!!!
"""
function solve(prob::CustomMonteCarloProblem; num_monte::Int=100, rel_transient_time::Real=0.5)

    #sol = []
    sol = @sync @distributed (vcat) for istep=1:num_monte

        sol_i = solve(prob.prob_func(prob.prob, istep, false))
        (___, N_t) = size(sol_i)

        transient_time = Int(round(rel_transient_time * N_t))

        sol_i = CustomSolution(sol_i[:,transient_time:end], sol_i.prob)
        res_i = prob.eval_func(sol_i, istep)

        if res_i[2]
            error("Problem signals 'repeat', but 'repeat' is not yet supported!")
        end
        #push!(sol, res_i[1])
        res_i[1]
    end

    return sol
end

"""
    CustomMCBBProblem <: MCBBProblem

Structure similar to `DEMCBBProblem` but for problems that can not be solved with `DifferentialEquations` as a backend.

Note that its supertypes are `MCBBProblem` and `myMCProblem`, but not any of the DifferentialEquations abstract problem types. Many functions that work on `DEMCBBProblem` work on `CustomMCBBProblem` as well as they are defined for `MCBBProblem`.

The struct has the following fields:
* `p`: `CustomMonteCarloProblem` to be solved, part of DifferentialEquations
* `N_mc`: Number of (Monte Carlo) runs to be solved
* `rel_transient_time`: Only after this time (relative to the total integration time) the solutions are evaluated
* `ic`: (``N_{mc} \\times N_{dim_{ic}}``)-Matrix containing initial conditions for each run.
* `par`: (``N_{mc} \\times N_{par}``)-Matrix containing parameter values for each run.
* `par_var`: `ParameterVar`, information about how the parameters are varied, see [`ParameterVar`](@ref)

# Constructors

It has the same constructors as [`DEMCBBProblem`](@ref) just with a `CustomProblem` instead of an `DEProblem`.

"""
struct CustomMCBBProblem <: MCBBProblem
    p::CustomMonteCarloProblem
    N_mc::Int64
    rel_transient_time::Float64
    ic::AbstractArray
    par::AbstractArray
    par_var::ParameterVar

    function CustomMCBBProblem(p::CustomProblem, ic_gens::Array{<:Function,1}, N_ic::Int, pars::DEParameters, par_range_tuple::ParameterVar, eval_ode_func::Function, tail_frac::Number)
        (ic_coupling_problem, ic, par, N_mc) = setup_ic_par_mc_problem(p, ic_gens, N_ic, pars, par_range_tuple)
        mcp = CustomMonteCarloProblem(p, ic_coupling_problem, eval_ode_func)
        new(mcp, N_mc, tail_frac, ic, par, par_range_tuple)
    end

    function CustomMCBBProblem(p::CustomProblem, ic_ranges::Array{<:AbstractArray,1}, pars::DEParameters, par_range_tuple::ParameterVar, eval_ode_func::Function, tail_frac::Number)
        (ic_coupling_problem, ic, par, N_mc) = setup_ic_par_mc_problem(p, ic_ranges, pars, par_range_tuple)
        mcp = CustomMonteCarloProblem(p, ic_coupling_problem, eval_ode_func)
        new(mcp, N_mc, tail_frac, ic, par, par_range_tuple)
    end

    # Direct Constructor
    CustomMCBBProblem(p::MonteCarloProblem, N_mc::Int64, rel_transient_time::Float64, ic::AbstractArray, par::AbstractArray, par_range_tuple::ParameterVar) = new(p, N_mc, rel_transient_time, ic, par, par_range_tuple)
end
CustomMCBBProblem(p::CustomProblem, ic_gens::Function, N_ic::Int, pars::DEParameters, par_range_tuple::ParameterVar, eval_ode_func::Function, tail_frac::Number) = CustomMCBBProblem(p, [ic_gens], N_ic, pars, par_range_tuple, eval_ode_func, tail_frac)
CustomMCBBProblem(p::CustomProblem, ic_gens::Union{Array{<:Function,1},Function}, N_ic::Int, pars::DEParameters, par_range_tuple::ParameterVar, eval_ode_func::Function) = CustomMCBBProblem(p,ic_gens,N_ic,pars,par_range_tuple,eval_ode_func, 0.9)

# automaticlly convert appropiate tuples to ParameterVar
CustomMCBBProblem(p::CustomProblem, ic_gens::Array{<:Function,1}, N_ic::Int, pars::DEParameters, par_range_tuple::Union{Tuple{Symbol,Union{AbstractArray,Function},<:Function},Tuple{Symbol,Union{AbstractArray,Function}}}, eval_ode_func::Function, tail_frac::Number) = CustomMCBBProblem(p,ic_gens, N_ic, pars, OneDimParameterVar(par_range_tuple...), eval_ode_func, tail_frac)
CustomMCBBProblem(p::CustomProblem, ic_ranges::Array{<:AbstractArray,1}, pars::DEParameters, par_range_tuple::Union{Tuple{Symbol,Union{AbstractArray,Function},<:Function},Tuple{Symbol,Union{AbstractArray,Function}}}, eval_ode_func::Function, tail_frac::Number) = CustomMCBBProblem(p, ic_ranges, pars, OneDimParameterVar(par_range_tuple...), eval_ode_func, tail_frac)
CustomMCBBProblem(p::CustomProblem, ic_gens::Function, N_ic::Int, pars::DEParameters, par_range_tuple::Union{Tuple{Symbol,Union{AbstractArray,Function},<:Function},Tuple{Symbol,Union{AbstractArray,Function}}}, eval_ode_func::Function, tail_frac::Number) = CustomMCBBProblem(p, ic_gens, N_ic, pars, OneDimParameterVar(par_range_tuple...), eval_ode_func, tail_frac)
CustomMCBBProblem(p::CustomProblem, ic_gens::Union{Array{<:Function,1},Function}, N_ic::Int, pars::DEParameters, par_range_tuple::Union{Tuple{Symbol,Union{AbstractArray,Function} ,<:Function},Tuple{Symbol,Union{AbstractArray,Function}}}, eval_ode_func::Function) = CustomMCBBProblem(p,ic_gens,N_ic,pars,OneDimParameterVar(par_range_tuple...),eval_ode_func)

"""
     define_new_problem(prob::CustomProblem, ic_par::AbstractArray, parameters::DEParameters, N_dim_ic::Int, ic_gens::AbstractArray, var_par::ParameterVar)

Helper functions that refurns a function returning new functions needed for `CustomMonteCarloProblem`.

"""
function define_new_problem(prob::CustomProblem, ic::AbstractArray, par::AbstractArray, parameters::DEParameters, N_dim_ic::Int, ic_gens::AbstractArray, var_par::ParameterVar)
    function new_problem(prob, i, repeat)
        _repeat_check(repeat, i, ic, ic_gens, N_dim_ic)
        CustomProblem(prob.f, ic[i,:], prob.tspan, var_par.new_par(parameters; _new_val_dict(var_par, par, N_dim_ic, i)...))
    end
end

"""
    solve(prob::CustomMCBBProblem)

Solves the `CustomMCBBProblem`.
"""
function solve(prob::CustomMCBBProblem)
    sol = solve(prob.p, num_monte=prob.N_mc, rel_transient_time=prob.rel_transient_time)
    N_t = length(sol[1])
    N_dim = length(prob.p.prob.u0)
    csol = CustomMCBBSolution(sol, prob.N_mc, N_t, N_dim, get_measure_dimensions(sol)..., (prob_i) -> solve(prob_i))
    sort!(csol, prob)
    return csol
end

"""
    CustomMCBBSolution <: MCBBSol

Type that stores the solutions of a `CustomMCBBProblem`. Is returned by the corresponding `solve` routine.

Its fields are:
* `sol`: solution of the `CustomMonteCarloProblem`
* `N_mc`: number of solutions saved / Monte Carlo trials runs
* `N_t`: number of time steps for each solutions
* `N_dim`: sytem dimension
* `N_meas`: number of measures used, ``N_{meas} = N_{meas_{dim}} + N_{meas_{global}}
* `N_meas_dim`: number of measures that are evalauted for every dimension seperatly
* `N_meas_global`: number of measures that are evalauted globally
* `solve_command`: A function that solves one individual run/trial with the same settings as where used to, for the `CustomProblem` type this is trivial (its the field `f`) but this field is here so that `CustomMCBBSolution` is compatible with `DEMCBBSol`

Note, in case `N_dim==1` => `N_meas_global == 0` and `N_meas_dim == N_meas`
"""
struct CustomMCBBSolution <: MCBBSol
    sol::AbstractArray
    N_mc::Int
    N_t::Int
    N_dim::Int
    N_meas::Int
    N_meas_dim::Int
    N_meas_global::Int
    solve_command::Function
end

"""
    normalize(sol::CustomMCBBSol, k::AbstractArray)

Returns a copy of `sol` with the solutions normalized to be in range [0,1]. It is possible to only select that some of the measures are normalized by providing an array with the indices of the measures that should be normalized, e.g. `[1,2]` for measure 1 and measure 2 to be normalized.

    normalize(sol::DEMCBBSol)

If no extra array is provided, all measures are normalized.
"""
function normalize(sol::CustomMCBBSolution, k::AbstractArray)
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

    sol_new = CustomMCBBSolution(new_mc_sol, sol.N_mc, sol.N_t, sol.N_dim, sol.N_meas, sol.N_meas_dim, sol.N_meas_global, sol.solve_command)
end
normalize(sol::CustomMCBBSolution) = normalize(sol, 1:sol.N_meas)
