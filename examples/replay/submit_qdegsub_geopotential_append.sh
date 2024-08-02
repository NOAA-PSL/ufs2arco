#!/bin/bash

#SBATCH -J append_qds_geo
#SBATCH -o slurm/append_qds_geo.%j.out
#SBATCH -e slurm/append_qds_geo.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=7
#SBATCH --partition=compute
#SBATCH -t 120:00:00

source /contrib/Tim.Smith/miniconda3/etc/profile.d/conda.sh
conda activate ufs2arco
python compute_geopotential.py \
    --input_path="gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.zarr" \
    --output_path="gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.zarr" \
    --num_workers=4
