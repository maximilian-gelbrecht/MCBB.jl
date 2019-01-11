# Running it on a HPC

The heavy-lifting in terms of parallelization is done by `MonteCarloProblem`/`solve` from DifferentialEquations, however we have to make most parts of the setup known to all processes with `@everywhere` and have a submit script.

Below, you will find an example of a script running a Kuramoto network and saving the results. Typically, (due to memory constraints) I would recommend to only solve the [`BifAnaMCProblem`](@ref) on the HPC, then save the solutions and do the remaining calculations (distance matrix and clustering) on your own PC.

For saving and loading data, I used `JLD2`. It needs all functions and variables in scope while loading objects that were used during the computation of the object. That's why I work with only one script that is used on both the remote and local, depending on whether the variable `cluster` is `true` or `false`.

## Julia Script (hpc_example.jl)

The Julia script is similar to the examples shown in the other guides in this documentation. Running Julia code in parallel on a HPC needs the `ClusterManagers` package to add the processes that are allocated by the regular Slurm script. If you are running this in parallel without a batch system, change `addprocs(ClusterManagers(...))` to `addprocs(...)`.

* For usage on the HPC set `cluster=true`
* For usage on your PC (to evaluate the results) set `cluster=false`

_DISCLAIMER_: In theory, not all of these `@everywhere`-commands should be needed, but somehow it was not working for me without them. The scripts below are tested on a HPC running SLURM for resource allocation.

```julia
cluster = true
using Distributed

if cluster
    using ClusterManagers
    N_tasks = parse(Int, ARGS[1])
    N_worker = N_tasks
    #addprocs(SlurmManager(N_worker))
    addprocs(2)
else
    using Plots
end
@everywhere using LightGraphs
using JLD2, FileIO, Clustering, StatsBase, Parameters
@everywhere using DifferentialEquations
@everywhere using Distributions
@everywhere using HighBifLib

@everywhere N = 40
@everywhere K = 0.5
@everywhere nd = Normal(0.5, 0.1) # distribution for eigenfrequencies # mean = 0.5Hz, std = 0.1Hz
@everywhere w_i_par = rand(nd,N)

@everywhere net = erdos_renyi(N, 0.2)
@everywhere A = adjacency_matrix(net)

@everywhere ic = zeros(N)
@everywhere ic_dist = Uniform(-pi,pi)
@everywhere kdist = Uniform(0,8)
@everywhere ic_ranges = ()->rand(ic_dist)
@everywhere N_ics = 10000
@everywhere K_range = ()->rand(kdist)
@everywhere pars = kuramoto_network_parameters(K, w_i_par, N, A)
@everywhere rp = ODEProblem(kuramoto_network, ic, (0.,2000.), pars)

@everywhere tail_frac = 0.9

@everywhere function eval_ode_run_kuramoto(sol, i)
    (N_dim, __) = size(sol)
    state_filter = collect(1:N)
    eval_funcs = [empirical_1D_KL_divergence_hist]
    global_eval_funcs = []
    eval_ode_run(sol, i, state_filter, eval_funcs, global_eval_funcs, cyclic_setback=true)
end

if cluster
    ko_mcp = BifAnaMCProblem(rp, ic_ranges, N_ics, pars, (:K, K_range), eval_ode_run_kuramoto, tail_frac)
    kosol = solve(ko_mcp)
    @save "kuramoto_sol.jld2" kosol ko_mcp
else
    @load "kuramoto_sol.jld2" kosol ko_mcp

    D = @time distance_matrix(kosol, parameter(ko_mcp), [1,0.75,0.5,1]);
    k = 4
    db_eps = median((KNN_dist_relative(D)))
    db_res = dbscan(D,db_eps,k)
    cluster_members = cluster_membership(ko_mcp,db_res,0.2,0.05);

    plot(cluster_members[1],cluster_members[2])
    savefig("kura-membership")
end
```

## Slurm Script

```bash
#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=highbiflib-test
#SBATCH --account=YOURACCOUNT
#SBATCH --output=highbiflib-test-%j-%N.out
#SBATCH --error=highbiflib-test-%j-%N.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --workdir=YOUR/WORK/DIR
#SBATCH --mail-type=END
#SBATCH --mail-user=YOUR@MAIL.ADDRESS

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

module load julia/1.0.2
module load hpc/2015
julia /path/to/the/script/hpc_example.jl $SLURM_NTASKS
```

The `module load ...` commands are for loading `julia` on the HPC that I use, there might be different kind of setups for your HPC.
