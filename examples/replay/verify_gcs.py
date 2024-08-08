import os
import subprocess
from datetime import timedelta
import pandas as pd
import xarray as xr
import numpy as np

from ufs2arco import FV3Dataset, Timer
from replay_mover import ReplayMoverQuarterDegree


def submit_slurm_checker(resolution, varname):

    the_code = \
        f"from verify_gcs import run_verification\n"+\
        f"run_verification('{resolution}', '{varname}')"

    slurm_dir = f"slurm/verify-{resolution}"
    txt = "#!/bin/bash\n\n" +\
        f"#SBATCH -J verify_{varname}\n"+\
        f"#SBATCH -o {slurm_dir}/{varname}.%j.out\n"+\
        f"#SBATCH -e {slurm_dir}/{varname}.%j.err\n"+\
        f"#SBATCH --nodes=1\n"+\
        f"#SBATCH --ntasks=1\n"+\
        f"#SBATCH --cpus-per-task=30\n"+\
        f"#SBATCH --partition=compute\n"+\
        f"#SBATCH -t 120:00:00\n\n"+\
        f"source /contrib/Tim.Smith/miniconda3/etc/profile.d/conda.sh\n"+\
        f"conda activate ufs2arco\n"+\
        f'python -c "{the_code}"'

    script_dir = "job-scripts"
    fname = f"{script_dir}/submit_{resolution}_{varname}_verification.sh"

    for this_dir in [slurm_dir, script_dir]:
        if not os.path.isdir(this_dir):
            os.makedirs(this_dir)

    with open(fname, "w") as f:
        f.write(txt)

    subprocess.run(f"sbatch {fname}", shell=True)


def get_missing_cycles(xda):

    miss = np.isnan(xda).load();
    missing_time = miss.time.where(miss, drop=True).values
    missing_cycles = missing_time + np.timedelta64(timedelta(hours=6))
    missing_cycles = [x for x in missing_cycles if pd.Timestamp(x).hour in [0, 6, 12, 18]]

    print("missing cycles")
    for mc in missing_cycles:
        print("    ", mc)

    print(f"... found {miss.sum().values} NaNs")
    print()
    return missing_cycles


def run_verification(resolution, varname):

    timer = Timer()

    timer.start(f"Checking {varname}")
    ds = xr.open_zarr(
        f"gcs://noaa-ufs-gefsv13replay/ufs-hr1/{resolution}/03h-freq/zarr/fv3.zarr",
        storage_options={"token": "anon"},
    )

    point2d = ds[varname].isel(grid_xt=100, grid_yt=100)
    if resolution in ("1.00-degree", "0.25-degree-subsampled"):
        point2d = point2d.load()
    all_missing_cycles = []
    if "pfull" in point2d.dims:

        for k in range(len(point2d.pfull)):
            print(f"Level {k}")
            missing_cycles = get_missing_cycles(point2d.isel(pfull=k))
            for mc in missing_cycles:
                if mc not in all_missing_cycles:
                    all_missing_cycles.append(mc)

    else:
        missing_cycles = get_missing_cycles(point2d)


    print(f"all missing cycles")
    for mc in all_missing_cycles:
        print(mc)

    timer.stop("Total Walltime: ")


if __name__ == "__main__":


    for resolution in ["1.00-degree", "0.25-degree-subsampled"]:
        ds = xr.open_zarr(
            f"gcs://noaa-ufs-gefsv13replay/ufs-hr1/{resolution}/03h-freq/zarr/fv3.zarr",
            storage_options={"token": "anon"},
        )
        data_vars = list(ds.data_vars)
        for key in data_vars:
            submit_slurm_checker(resolution, key)
