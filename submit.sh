#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=roess-net-repeat
#SBATCH --account=brasnet
#SBATCH --output=roess-net-repeat-%j-%N.out
#SBATCH --error=roess-net-repeat-%j-%N.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
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
julia /p/tmp/maxgelbr/code/HighBifLib.jl/run_mc.jl $SLURM_NTASKS
