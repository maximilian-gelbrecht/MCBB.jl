# Custom Problem

```@docs
CustomProblem
solve(prob::CustomProblem)
CustomSolution
solve(prob::CustomMonteCarloProblem; num_monte::Int, rel_transient_time::Real)
CustomMonteCarloProblem
solve(prob::CustomMCBBProblem)
CustomMCBBProblem
CustomMCBBSolution
normalize(sol::CustomMCBBSolution, k::AbstractArray)
```
