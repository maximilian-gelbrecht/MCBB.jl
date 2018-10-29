using DifferentialEquations
using Parameters
import DifferentialEquations.solve
import DifferentialEquations.problem_new_parameters

### Introduces a Problem type for Bifurcation analysis
# sets up a DEProblem, solves it, then changes a parameter and sets the endpoint
# of the previous to be IC of a new problem. repeats for a specified amount of time
# times
#
# Arguments:
#       p :: Base DifferentialEquations Problem
#       par_range :: Description of how the parameter should be changed
#               is a tuple of:
#                    -   name of the parameter as a symbol
#                    -   AbstractArray or Function that contains all parameters for the experiment or a function that generates parameter values, the function has to be format (oldvalue) -> (newvalue)
#                   - OPTIONAL: a function that maps (old_parameter_instance; (par_range[1],new_parameter_value)) -> new_parameter_instance. Default is 'reconstruct' from @with_kw/Parameters.jl is used
#       N :: OPTIONAL
#       eval_ode_run :: function, same as for MonteCarloProblems
#       ic_bounds:: Bounds for the IC that should not be exceeded
#       par_bounds:: Bounds for the parameter that should not be exceeded
#       hard_bounds:: If a bound is reached: if true, stops the iteration, false continous with (upper/lower bound as IC)

struct BifAnalysisProblem
    prob::DEProblem # Base DifferentialEquations Problem
    par_range::Tuple{Symbol,Union{AbstractArray,Function},<:Function}
    N::Int64
    eval_func::Function
    ic_bounds::AbstractArray
    par_bounds::AbstractArray
    hard_bounds::Bool
    par::AbstractArray

    # different constructors for different kinds of par_range tuples...

    # 1. tuple has array
    function BifAnalysisProblem(p::DEProblem, par_range::Union{Tuple{Symbol,AbstractArray}, Tuple{Symbol,AbstractArray,<:Function}}, eval_func::Function, ic_bounds::AbstractArray=[-Inf,Inf], par_bounds::AbstractArray=[-Inf,Inf], hard_bounds::Bool=false)
        par_range = _var_par_check(par_range)
        N = length(par_range[2])
        par_vector = compute_parameters(p, par_range, N)
        new(p, par_range, N, eval_func, ic_bounds, par_bounds, hard_bounds, par_vector)
    end

    # 2. tuple has function
    function BifAnalysisProblem(p::DEProblem, par_range::Union{Tuple{Symbol,<:Function,<:Function},Tuple{Symbol,<:Function}}, N::Integer, eval_func::Function, ic_bounds::AbstractArray=[-Inf,Inf], par_bounds::AbstractArray=[-Inf,Inf], hard_bounds::Bool=false)
        par_range = _var_par_check(par_range)
        par_vector = compute_parameters(p, par_range, N)
        BifAnalysisProblem(p, par_range, N, eval_func, ic_bounds, par_bounds, hard_bounds, par_vector)
    end

    # direct constructor
    function BifAnalysisProblem(p::DEProblem, par_range::Tuple{Symbol,Union{AbstractArray,Function},<:Function}, N::Int64, eval_func::Function, ic_bounds::AbstractArray=[-Inf,Inf],par_bounds::AbstractArray=[-Inf,Inf], hard_bounds::Bool=false)
        par_vector = compute_parameters(p, par_range, N)
        new(p, par_range, N, eval_func, ic_bounds, par_bounds, hard_bounds, par_vector)
    end
end
parameter(prob::BifAnalysisProblem) = prob.par_vector

struct BifAnalysisSolution <: myMCSol
    sol::AbstractArray
    par_vec::AbstractArray
    N_mc::Int
end

# computes the parameters that are used for the calculation
function compute_parameters(p::DEProblem, par_range::Union{Tuple{Symbol, Union{AbstractArray,<:Function},<:Function}, Tuple{Symbol,Union{AbstractArray,<:Function}}}, N::Integer)
    if typeof(par_range[2])<:Function
        par_vector = zeros(N)
        par_vector[1] = getfield(p.p,par_range[1]) # IC of first DEProblem

        for istep=2:prob.N
            par_vector[istep] = p.par_range[2](par_vector[istep-1])
        end
    else
        par_vector = par_range[2]
    end
    par_vector
end

# custom solve for the BifAnalysisProblem.
# Saves and evaluates only after transient at a constant step size
function solve(prob::BifAnalysisProblem, N_t=400::Int, rel_transient_time::Float64=0.5, kwargs...)
    t_save = collect(tsave_array(prob.prob, N_t, rel_transient_time))

    par_vector = prob.par_vector

    solve_command(prob_in) = solve(prob_in, dense=false, save_everystep=false, saveat=t_save, savestart=false; kwargs...)

    sol_i = solve_command(prob.prob)
    sol = []
    push!(sol,eval_ode_run(sol_i, 1)[1])
    for istep=2:prob.N
        # bounds check for new par
        if (par_vector[istep] < prob.par_bounds[1]) | (par_vector[istep] > prob.par_bounds[2])
            if prob.hard_bounds
                return BifAnalysisSolution(sol,par_vector[1:istep], N_t)
            else
                if (par_vector[istep] < prob.par_bounds[1])
                    par_vector[istep] = prob.par_bounds[1]
                else
                    par_vector[istep] = prob.par_bounds[2]
                end
            end
        end

        deprob = problem_new_parameters(prob.prob, prob.par_range[3](prob.prob.p; (prob.par_range[1], par_vector[istep])))

        new_u0 = sol_i[end]

        # bounds check of new IC
        if (sum(new_u0 .< prob.ic_bounds[1])>0) | (sum(new_u0 .> prob.ic_bounds[2])<0)
            if prob.hard_bounds
                return BifAnalysisSolution(sol,par_vector[1:step], N_t)
            else
                if (sum(new_u0 .< prob.ic_bounds[1])>0) < prob.ic_bounds[1]
                    new_u0[new_u0 .< prob.ic_bounds[1]] = prob.ic_bounds[1]
                else
                    new_u0[new_u0 .> prob.ic_bounds[2]] = prob.ic_bounds[2]
                end
            end
        end

        deprob = remake(deprob, u0=new_u0)
        sol_i = solve_command(deprob)
        push!(sol,eval_ode_run(sol_i, istep)[1])
    end

    return BifAnalysisSolution(sol, par_vector, N_t)
end

# weirdly enough there is a problem_new_parameters routine in DiffEqBase for all problems types EXCEPT for discrete problems
function problem_new_parameters(prob::DiscreteProblem,p;kwargs...)
  uEltype = eltype(p)
  u0 = [uEltype(prob.u0[i]) for i in 1:length(prob.u0)]
  tspan = (uEltype(prob.tspan[1]),uEltype(prob.tspan[2]))
  DiscreteProblem{isinplace(prob)}(prob.f,u0,tspan,p;
  callback = prob.callback,
  kwargs...)
end
