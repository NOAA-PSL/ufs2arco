import os
import subprocess

from ufs2arco import FV3Dataset, Timer
from replay_mover import ReplayMoverQuarterDegree


def submit_slurm_filler(job_id, mover):

    the_code = \
        f"from replay_mover import ReplayMoverQuarterDegree\n"+\
        f"from replay_filler import ReplayFiller\n"+\
        f"mover = ReplayMoverQuarterDegree(\n"+\
        f"    n_jobs={mover.n_jobs},\n"+\
        f"    config_filename='{mover.config_filename}',\n"+\
        f"    storage_options={mover.storage_options},\n"+\
        f"    main_cache_path='{mover.main_cache_path}',\n"+\
        f")\n"+\
        f"filler = ReplayFiller(mover=mover)\n"+\
        f"filler.run({job_id})"

    slurm_dir = "slurm/replay-fill-0.25-degree"
    txt = "#!/bin/bash\n\n" +\
        f"#SBATCH -J rfqd{job_id:03d}\n"+\
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
    fname = f"{script_dir}/submit_qfiller{job_id:03d}.sh"

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

    # Recreated from what we used to put up quarter degree in the first place
    mover = ReplayMoverQuarterDegree(
        n_jobs=1,
        config_filename="config-0.25-degree.yaml",
        storage_options={"token": "/contrib/Tim.Smith/.gcs/replay-service-account.json"},
        main_cache_path="/lustre/Tim.Smith/tmp-replay/0.25-degree",
    )

    localtime.start("Run slurm jobs")
    for job_id in range(mover.n_jobs):
        submit_slurm_filler(job_id, mover)
    localtime.stop()

    walltime.stop("Walltime Time")
