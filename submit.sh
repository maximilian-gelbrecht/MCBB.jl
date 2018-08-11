#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=julia-test2
#SBATCH --account=brasnet
#SBATCH --output=julia-test2-%j-%N.out
#SBATCH --error=julia-test2-%j-%N.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --workdir=/p/tmp/maxgelbr
#SBATCH --mail-type=END
#SBATCH --mail-user=gelbrecht@pik-potsdam.de
#SBATCH --time 01:00:00

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

module load julia/0.6.2
module load hpc/2015
julia /p/tmp/maxgelbr/julia-test/montecarlo/run_mc.jl $SLURM_NTASKS
