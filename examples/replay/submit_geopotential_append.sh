#!/bin/bash

#SBATCH -J append_geopotential
#SBATCH -o slurm/append_geopotential.%j.out
#SBATCH -e slurm/append_geopotential.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=10
#SBATCH --partition=compute
#SBATCH -t 120:00:00

source /contrib/Tim.Smith/miniconda3/etc/profile.d/conda.sh
conda activate ufs2arco

python append_geopotential.py \
    --input_path="gs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/03h-freq/zarr/fv3.zarr" \
    --output_path="gs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/03h-freq/zarr/fv3.zarr" \
    --num_workers=4

python append_geopotential.py \
    --input_path="gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.zarr" \
    --output_path="gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.zarr" \
    --num_workers=4
