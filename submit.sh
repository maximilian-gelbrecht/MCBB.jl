#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=log-bif
#SBATCH --account=brasnet
#SBATCH --output=log-bif-%j-%N.out
#SBATCH --error=log-bif-%j-%N.err
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=8
#SBATCH --workdir=/p/tmp/maxgelbr
#SBATCH --mail-type=END
#SBATCH --mail-user=gelbrecht@pik-potsdam.de

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

module load julia/0.6.2
module load hpc/2015
julia /p/tmp/maxgelbr/code/HighBifLib.jl/run_mc.jl $SLURM_NTASKS
