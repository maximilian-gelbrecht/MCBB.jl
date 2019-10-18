using DifferentialEquations
using Parameters
import DifferentialEquations.solve

"""
    ContinuationProblem <: myMCProblem

Introduces a Problem type for Bifurcation analysis. It sets up a DEProblem, solves it, then changes a parameter and sets the endpoint of the previous to be IC of a new problem. Repeats for a specified amount of times

# Constructor

    ContinuationProblem(p::DiffEqBase.DEProblem, par_range::ParameterVar, N::Int64, eval_func::Function, ic_bounds::AbstractArray=[-Inf,Inf],par_bounds::AbstractArray=[-Inf,Inf], hard_bounds::Bool=false)

* `p`: Base DifferentialEquations Problem
* `par_range`: Description of how the parameter should be changed, `ParameterVar` OR a tuple of (First: name of the parameter as a symbol, Second: AbstractArray or Function that contains all parameters for the experiment or a function that generates parameter values. The function has to be format `(oldvalue) -> (newvalue)`, Third: OPTIONAL: a function that maps `(old_parameter_instance; (par_range[1],new_parameter_value)) -> new_parameter_instance`. Default is 'reconstruct' from @with_kw/Parameters.jl is used)
* `N`: OPTIONAL, Only needed if parameter variation is given a function to generate new parameter values. If so, `N` is the total number of parameters and thus DEProblems to be solved
* `eval_ode_run`: function, same as for `EnsembleProblems`
* `ic_bounds`: Bounds for the IC that should not be exceeded
* `par_bounds`: Bounds for the parameter that should not be exceeded
* `hard_bounds`: If a bound is reached: if true, stops the iteration, false continous with (upper/lower bound as IC)

Fields of the struct are the same as the arguments of the constructor.
"""
struct ContinuationProblem <: myMCProblem
    prob::DiffEqBase.DEProblem # Base DifferentialEquations Problem
    par_range::OneDimParameterVar
    N::Int64
    eval_func::Function
    ic_bounds::AbstractArray
    par_bounds::AbstractArray
    hard_bounds::Bool
    par::AbstractArray

    function ContinuationProblem(p::DiffEqBase.DEProblem, par_range::OneDimParameterVar, N::Int64, eval_func::Function, ic_bounds::AbstractArray=[-Inf,Inf],par_bounds::AbstractArray=[-Inf,Inf], hard_bounds::Bool=false)
        par_vector = compute_parameters(p, par_range, N)
        N = length(par_vector) # updated N in case a boundary is hit
        new(p, par_range, N, eval_func, ic_bounds, par_bounds, hard_bounds, par_vector)
    end

    # direct constructor
    function ContinuationProblem(p::DiffEqBase.DEProblem, par_range::OneDimParameterVar, N::Int64, eval_func::Function, ic_bounds::AbstractArray,par_bounds::AbstractArray, hard_bounds::Bool, par_vector::AbstractArray)
        new(p, par_range, N, eval_func, ic_bounds, par_bounds, hard_bounds, par_vector)
    end
end

ContinuationProblem(p::DiffEqBase.DEProblem, par_range::Union{Tuple{Symbol, <:Function,<:Function}, Tuple{Symbol,<:Function}}, N::Int, eval_func::Function, ic_bounds::AbstractArray=[-Inf,Inf], par_bounds::AbstractArray=[-Inf,Inf], hard_bounds::Bool=false) = ContinuationProblem(p, OneDimParameterVar(par_range...), N, eval_func, ic_bounds, par_bounds, hard_bounds)
ContinuationProblem(p::DiffEqBase.DEProblem, par_range::Union{Tuple{Symbol, AbstractArray,<:Function}, Tuple{Symbol,AbstractArray}}, eval_func::Function, ic_bounds::AbstractArray=[-Inf,Inf], par_bounds::AbstractArray=[-Inf,Inf], hard_bounds::Bool=false) = ContinuationProblem(p, OneDimParameterVar(par_range...), length(par_range[2]), eval_func, ic_bounds, par_bounds, hard_bounds)

parameter(prob::ContinuationProblem) = prob.par

"""
    ContinuationSolution <: myMCSol

Solution object that is returned by `solve(prob::ContinuationProblem,...)`.

Fields are:
* `sol`: Array of Arrays, analogously to `EnsembleSolution`
* `par`: Array, parameters of all runs
* `N_mc`: Number of runs / DEProblems solved
* `N_meas`: number of measures used, ``N_{meas} = N_{meas_{dim}} + N_{meas_{global}}``
* `N_meas_dim`: number of measures that are evalauted for every dimension seperatly
* `N_meas_global`: number of measures that are evalauted globally
* `N_meas_matrix`: number of measures that return matrices.
"""
struct ContinuationSolution <: myMCSol
    sol::AbstractArray
    par::AbstractArray
    N_mc::Int
    N_meas::Int # number of measures used, N_meas = N_meas_dim + N_meas_global
    N_meas_dim::Int # number of measures that are evaluated for every dimension
    N_meas_global::Int # number of global measures, in the case of system dimension == 1, N_meas_global = 1 = N_meas and N_meas_dim = 0
    N_meas_matrix::Int
end

"""
    compute_parameters(p::DiffEqBase.DEProblem , par_range::OneDimParameterVar, N::Integer)

Computes the parameters that are used for the calculation and returns them as an array.
"""
compute_parameters(p::DiffEqBase.DEProblem , par_range::ParameterVarArray, N::Integer) = par_range.arr
function compute_parameters(p::DiffEqBase.DEProblem , par_range::ParameterVarFunc, N::Integer)
    par_vector = zeros(N)
    par_vector[1] = getfield(p.p,par_range.name) # IC of first DEProblem

    for istep=2:N
        par_vector[istep] = par_range.new_val(par_vector[istep-1])
    end
    par_vector
end

"""
     solve(prob::ContinuationProblem, N_t=400::Int, rel_transient_time::Float64=0.9; return_probs::Bool=false, reltol::Float64=1e-9, cyclic_ic::Bool=false, kwargs...)

Custom solve for the ContinuationProblem. Saves and evaluates only after transient at a constant step size.

* `N_t` - Int, Number of timesteps of each solution of a DEProblem
* `rel_transient_time` - Percentage of time after which the solutions are evaluated
* `return_probs` - if 'true' returns a array of DEProblems that were solved
* `reltol` - relative tolerance of the solver.Eespacially for systems with constantly growing variables such as certain phase oscilattors the tolerence has to be very small
* `cyclic_ic` - if true the initial conditions are always within ``[-\\pi,\\pi]``
"""
function solve(prob::ContinuationProblem, N_t=400::Int, rel_transient_time::Float64=0.9; return_probs::Bool=false, reltol::Float64=1e-9, cyclic_ic::Bool=false, kwargs...)
    t_save = collect(tsave_array(prob.prob, N_t, rel_transient_time))

    par_vector = prob.par

    solve_command(prob_in) = solve(prob_in, dense=false, reltol=reltol, save_everystep=false, saveat=t_save, savestart=false; kwargs...)

    sol_i = solve_command(prob.prob)
    sol = []
    push!(sol,prob.eval_func(sol_i, 1)[1])

    N_dim , ___ = size(sol_i)

    if return_probs
        prob_vec = []
        push!(prob_vec,prob.prob)
    end

    for istep=2:prob.N
        # bounds check for new par
        if (par_vector[istep] < prob.par_bounds[1]) | (par_vector[istep] > prob.par_bounds[2])
            if prob.hard_bounds
                par_vector = par_vector[1:istep]
                break
            else
                if (par_vector[istep] < prob.par_bounds[1])
                    par_vector[istep] = prob.par_bounds[1]
                else
                    par_vector[istep] = prob.par_bounds[2]
                end
            end
        end

        deprob = remake(prob.prob, p=prob.par_range.new_par(prob.prob.p; Dict(prob.par_range.name => par_vector[istep])...))

        new_u0 = sol_i[end]

        if cyclic_ic
            new_u0 = mod.(new_u0,2pi)
            new_u0[new_u0 .> pi] -= 2pi
        end

        # bounds check of new IC
        if (sum(new_u0 .< prob.ic_bounds[1])>0) | (sum(new_u0 .> prob.ic_bounds[2])<0)
            if prob.hard_bounds
                par_vector = par_vector[1:step]
                break
            else
                if (sum(new_u0 .< prob.ic_bounds[1])>0) < prob.ic_bounds[1]
                    new_u0[new_u0 .< prob.ic_bounds[1]] = prob.ic_bounds[1]
                else
                    new_u0[new_u0 .> prob.ic_bounds[2]] = prob.ic_bounds[2]
                end
            end
        end

        deprob = remake(deprob, u0=new_u0)
        if return_probs
            push!(prob_vec,deprob)
        end
        sol_i = solve_command(deprob)
        push!(sol,prob.eval_func(sol_i, istep)[1])
    end

    if return_probs
        return (ContinuationSolution(sol, par_vector, length(par_vector), get_measure_dimensions(sol, N_dim)...), prob_vec)
    else
        return ContinuationSolution(sol, par_vector, prob.N, get_measure_dimensions(sol, N_dim)...)
    end
end
