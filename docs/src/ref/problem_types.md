# Problem Types

```@docs
myMCProblem
BifAnaMCProblem
parameter(p::BifAnaMCProblem)
BifMCSol
sort(sol::BifMCSol, prob::BifAnaMCProblem, i::Int=1)
sort!(sol::BifMCSol, prob::BifAnaMCProblem, i::Int=1)
sort(prob::BifAnaMCProblem, i::Int=1)
sort!(prob::BifAnaMCProblem, i::Int=1)
show_results
get_measure
normalize(sol::BifMCSol, k::AbstractArray)
solve(prob::BifAnaMCProblem, alg=nothing, N_t=400::Int, parallel_type=:parfor; flag_check_inf_nan=true, custom_solve::Union{Function,Nothing}=nothing, kwargs...)
```

## (Semi) Internal functions

The following functions are usually not needed by the user as they are called automatically. They are still exported as they can be useful for some custom cases.

```@docs
setup_ic_par_mc_problem
define_new_problem
tsave_array
```
