"""We can't use a dask cluster, because it cannot serialize the tasks of opening multiple
datasets with an io buffered reader object, orsomething.

So, this is easy enough to just submit separate slurm jobs that work on their own job ID.
"""

import os
import subprocess

from ufs2arco import FV3Dataset, Timer
from replay_mover import ReplayMover1Degree


def submit_slurm_mover(job_id, mover):

    the_code = \
        f"from replay_mover import ReplayMover1Degree\n"+\
        f"mover = ReplayMover1Degree(\n"+\
        f"    n_jobs={mover.n_jobs},\n"+\
        f"    config_filename='{mover.config_filename}',\n"+\
        f"    storage_options={mover.storage_options},\n"+\
        f"    main_cache_path='{mover.main_cache_path}',\n"+\
        f"    component='{mover.component}',\n"+\
        f")\n"+\
        f"mover.run({job_id})"

    slurm_dir = "slurm/replay-1.00-degree"
    txt = "#!/bin/bash\n\n" +\
        f"#SBATCH -J r1d{job_id:03d}\n"+\
        f"#SBATCH -o {slurm_dir}/{job_id:03d}.%j.out\n"+\
        f"#SBATCH -e {slurm_dir}/{job_id:03d}.%j.err\n"+\
        f"#SBATCH --nodes=1\n"+\
        f"#SBATCH --ntasks=1\n"+\
        f"#SBATCH --cpus-per-task=30\n"+\
        f"#SBATCH --partition=compute\n"+\
        f"#SBATCH -t 120:00:00\n\n"+\
        f"source /contrib/Tim.Smith/miniconda3/etc/profile.d/conda.sh\n"+\
        f"conda activate ufs2arco\n"+\
        f'python -c "{the_code}"'

    script_dir = "job-scripts"
    fname = f"{script_dir}/submit_1mover{job_id:03d}.sh"

    for this_dir in [slurm_dir, script_dir]:
        if not os.path.isdir(this_dir):
            os.makedirs(this_dir)

    with open(fname, "w") as f:
        f.write(txt)

    subprocess.run(f"sbatch {fname}", shell=True)


if __name__ == "__main__":

    walltime = Timer()
    localtime = Timer()

    walltime.start("Initializing job")

    mover = ReplayMover1Degree(
        n_jobs=15,
        config_filename="config-1.00-degree.yaml",
        storage_options={"token": "/contrib/Tim.Smith/.gcs/replay-service-account.json"},
        main_cache_path="/lustre/Tim.Smith/tmp-replay/1.00-degree",
        component="fv3",
    )

    localtime.start("Make and Store Container Dataset")
    mover.store_container()
    localtime.stop()

    localtime.start("Run slurm jobs")
    for job_id in range(mover.n_jobs):
        submit_slurm_mover(job_id, mover)
    localtime.stop()

    walltime.stop("Walltime Time")
