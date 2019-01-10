# Running it on a HPC

The heavy-lifting in terms of parallelization is done by `MonteCarloProblem`/`solve` from DifferentialEquations, however we have to make most parts of the setup known to all processes with `@everywhere` and have a submit script.

_DISCLAIMER_: In theory, not all of these `@everywhere`-commands should be needed, but somehow it was not working for me without them. The scripts below are tested on a HPC running SLURM for resource allocation.

Below, you will find an example of a script running a Kuramoto network and saving the results. Typically, (due to memory constraints) I would recommend to only solve the [`BifAnaMCProblem`](@ref) on the HPC, then save the solutions and due the remaining calculations (distance matrix and clustering) on your own PC.

For saving and loading data, I used `JLD2`. It needs all functions and variable in scope while loading objects that were used during the computation of the object. That's why I work with only one script that used on both the remote and local, depending on whether the variable `cluster` is defined or not.

## Julia Script

The Julia script is similar to the examples shown in the other guides in this documentation. 
