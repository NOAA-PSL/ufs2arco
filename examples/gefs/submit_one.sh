#!/bin/bash

#SBATCH -J gefs-1.00
#SBATCH -o /global/cfs/cdirs/m4718/timothys/gefs/one-degree/slurm/%j.out
#SBATCH -e /global/cfs/cdirs/m4718/timothys/gefs/one-degree/slurm/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=8
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 06:00:00

conda activate /global/common/software/m4718/timothys/graphufs
export PYTHONPATH=~/ufs2arco
srun python -c "from ufs2arco.driver import Driver; Driver('recipe.one.yaml').run()"
