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
#

struct BifAnalysisProblem
    prob::DEProblem # Base DifferentialEquations Problem
    par_range::Tuple{Symbol,Union{AbstractArray,Function},<:Function}
    N::Int64
    eval_func::Function

    # different constructors for different kinds of par_range tuples...
    function BifAnalysisProblem(p::DEProblem, par_range::Tuple{Symbol,Union{AbstractArray,Function}},N::Int64, eval_func::Function)
        par_range_tuple = (par_range[1], par_range[2], reconstruct)
        new(p,par_range_tuple,N,eval_func)
    end

    function BifAnalysisProblem(p::DEProblem, par_range::Union{Tuple{Symbol,Union{AbstractArray},<:Function},Tuple{Symbol,Union{AbstractArray}}}, eval_func::Function)
        N = length(par_range[2])
        BifAnalysisProblem(p,par_range,N, eval_func)
    end

    function BifAnalysisProblem(p::DEProblem, par_range::Tuple{Symbol,Union{AbstractArray,Function},<:Function}, N::Int64, eval_func::Function)
        new(p,par_range,N,eval_func)
    end
end

# custom solve for the BifAnalysisProblem.
# Saves and evaluates only after transient at a constant step size
function solve(prob::BifAnalysisProblem, N_t=400::Int, rel_transient_time::Float64=0.5, kwargs...)
    t_save = collect(tsave_array(prob.prob, N_t, rel_transient_time))

    # parameter vector
    if typeof(prob.par_range[2])<:Function

        par_vector = zeros(prob.N)
        par_vector[1] = prob.prob.p

        for istep=2:prob.N
            par_vector[istep] = par_range[2](par_vector[istep-1])
        end
    else
        par_vector = prob.par_range[2]
    end

    solve_command(prob_in) = solve(prob_in, dense=false, save_everystep=false, saveat=t_save, savestart=false; kwargs...)

    sol_i = solve_command(prob.prob)
    sol = []
    push!(sol,eval_ode_run(sol_i, 1)[1])
    for istep=2:prob.N

        deprob = problem_new_parameters(prob.prob, prob.par_range[3](prob.prob.p; (prob.par_range[1], par_vector[istep])))
        deprob = remake(deprob, u0=sol_i[end])
        sol_i = solve_command(deprob)
        push!(sol,eval_ode_run(sol_i, istep)[1])
    end

    return (sol, par_vector)
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
