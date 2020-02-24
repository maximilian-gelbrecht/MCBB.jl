# Problem Types

```@docs
myMCProblem
MCBBProblem
DEMCBBProblem
parameter(p::MCBBProblem, i::Int; complex_returns_abs=true)
MCBBSol
DEMCBBSol
sort(sol::MCBBSol, prob::MCBBProblem, i::Int=1)
sort!(sol::MCBBSol, prob::MCBBProblem, i::Int=1)
sort(prob::MCBBProblem, i::Int=1)
sort!(prob::MCBBProblem, i::Int=1)
show_results
get_measure
normalize(sol::DEMCBBSol, k::AbstractArray)
solve(prob::DEMCBBProblem, alg=nothing, N_t=400::Int, parallel_type=:parfor; flag_check_inf_nan=true, custom_solve::Union{Function,Nothing}=nothing, kwargs...)
MCBB.save(p::DEMCBBProblem, file_name::String)
MCBB.load_prob
MCBB.load_sol
```

## (Semi) Internal functions

The following functions are usually not needed by the user as they are called automatically. They are still exported as they can be useful for some custom cases.

```@docs
setup_ic_par_mc_problem
define_new_problem
tsave_array
```
