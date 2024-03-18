#!/bin/bash

#SBATCH -J quarter_container
#SBATCH -o slurm/quarter_container.%j.out
#SBATCH -e slurm/quarter_container.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=compute
#SBATCH -t 120:00:00

source /contrib/Tim.Smith/miniconda3/etc/profile.d/conda.sh
conda activate ufs2arco
python move_quarter_degree.py
