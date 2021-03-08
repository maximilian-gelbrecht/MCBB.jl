using Distributed
import DifferentialEquations.solve

"""
    SolverParameterVar(name::Symbol, new_val)

* `name`: Symbol of the name of the parameter
* `new_val`:: Function that returns a new value, signature: `(i::Int) -> new_value::Number` or ()->new_value::Number or Array with all parameter values

Internally there is an seperate struct for the cases that either a function that generates random values or the values directly are handed over / saved.
"""
struct SolverParameterVarFunc <: AbstractSolverParameterVar
    name::Symbol # keyword
    new_val::Function

    function SolverParameterVarFunc(name::Symbol, new_val::Function)
        new_val = verify_func(new_val)
        return new(name, new_val)
    end
end

struct SolverParameterVarArray <: AbstractSolverParameterVar
    name::Symbol
    new_val::Function
    N::Integer
    arr::AbstractArray
end

function SolverParameterVar(name::Symbol, arr::AbstractArray)
    function new_val(i)
        arr[i]
    end
    SolverParameterVarArray(name,new_val,length(collect(arr)),arr)
end
SolverParameterVar(name::Symbol, new_val::Function) = SolverParameterVarFunc(name, new_val)

"""
    length(parvar::AbstractSolverParameterVar)

Length returns the amount of Parameters that are setup to be varied.
"""
Base.length(parvar::AbstractSolverParameterVar) = 1

"""
    SolverVarEnsembleProblem{T}

SolverVarEnsembleProblem that allows solver parameters to be varied, otherwise similar in behaviour to that of DifferentialEquations.jl

Fields and Constructor:

* `prob::T`: Same as for EnsembleProblem
* `prob_func::Function`: Same as for EnsembleProblem
* `eval_func::Function`: Same as for EnsembleProblem
* `parvar::SolverParameterVar`
* `par::AbstractArray`: Values of the parameter

"""
struct SolverVarEnsembleProblem{T}
    prob::T
    prob_func::Function
    eval_func::Function
    parvar::AbstractSolverParameterVar
    par::AbstractArray
end


"""
    solve(prob::SolverVarEnsembleProblem, alg=nothing, parallel_type=nothing; trajectories::Int=100, rel_transient_time::Real=0.5, kwargs...)

Solves the [`SolverVarEnsembleProblem`](@ref) `num_monte`-times. Allows for variation of solver parameters. Hands kwargs over to the `solve`.

Parallel_type is not used (it is always pfor parallelized), but is an argument to have the same signature as the call for the regular EnsembleProblem

"""
function solve(prob::SolverVarEnsembleProblem, alg=nothing, parallel_type=nothing; trajectories::Int=100, rel_transient_time::Real=0.5, kwargs...)

    sol = @sync @distributed (vcat) for istep=1:trajectories

        sol_i = solve(prob.prob_func(prob.prob, istep, false), alg; prob.parvar.name => prob.par[istep], kwargs...)

        res_i = prob.eval_func(sol_i, istep)

        if res_i[2]
            error("Problem signals 'repeat', but 'repeat' is not yet supported!")
        end
        res_i[1]
    end

    return sol
end


"""
    define_new_problem(prob, ic::Abstract, par::AbstractArray, parameters::DEParameters, N_dim_ic::Int, ic_gens::AbstractArray, var_par::SolverParameterVar)

Returns the functions that returns new DifferentialEquations problems needed for `EnsembleProblem`. For the 'SolverParameterVar' Problems it only changes the initial conditions and not the paremters of the problem.

* `prob`: A Problem from DifferentialEquations, currently supported are `DiscreteProblem`, `ODEProblem`, `SDEProblem`, `DDEProblem` the base problem one is interested in.
* `ic`: (``N_{mc} \\times N_{dim_{ic}}``)-Matrix containing initial conditions for each run.
* `par`: (``N_{mc} \\times N_{par}``)-Matrix containing parameter values for each run. Left Constant or Blank here, but kept for consitency with the other ['define_new_problem'](@ref) calls.
* `N_dim_ic`: system dimension
* `ic_gens`: Array of functions or arrays/ranges that contain/generate the ICs.
* `var_par`:  `SOlverParameterVar`, information about how the parameters are varied, see [`SolverParameterVar`](@ref).
"""
function define_new_problem(prob, ic::AbstractArray, par::AbstractArray, parameters::DEParameters, N_dim_ic::Int, ic_gens::AbstractArray, var_par::AbstractSolverParameterVar)
    function new_problem(prob, i, repeat)
        _repeat_check(repeat, i, ic, ic_gens, N_dim_ic)
        remake(prob; u0=ic[i,:])
    end
    new_problem
end
define_new_problem(prob, ic::AbstractArray, par::AbstractArray, parameters::DEParameters, N_dim_ic::Int, var_par::AbstractSolverParameterVar) = (prob, i, repeat) -> remake(prob; u0=ic[i,:])
