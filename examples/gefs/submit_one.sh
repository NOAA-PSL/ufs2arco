#!/bin/bash

#SBATCH -J gefs-1.00
#SBATCH -o /global/cfs/cdirs/m4718/timothys/gefs/one-degree/slurm/one_degree.%j.out
#SBATCH -e /global/cfs/cdirs/m4718/timothys/gefs/one-degree/slurm/one_degree.%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=2
#SBATCH --qos=debug
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 00:30:00

conda activate /global/common/software/m4718/timothys/graphufs
export PYTHONPATH=~/ufs2arco
srun python -c "from ufs2arco.driver import Driver; Driver('recipe.one.yaml').run()"
