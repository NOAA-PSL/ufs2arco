from mpi4py import MPI
import logging
import pandas as pd

from ufs2arco.mpi import MPITopology
from ufs2arco.gefsdataset import GEFSDataset
from ufs2arco.batchloader import BatchLoader, MPIBatchLoader

if __name__ == "__main__":

    topo = MPITopology(log_dir="/global/cfs/cdirs/m4718/timothys/gefs/one-degree/logs")
    gefs = GEFSDataset("/global/homes/t/timothys/ufs2arco/examples/gefs/recipe.yaml")

    loader = MPIBatchLoader(
        dataset=gefs,
        batch_size=4,
        num_workers=0,
        max_queue_size=1,
        mpi_topo=topo,
        cache_dir="/pscratch/sd/t/timothys/gefs",
    )

    if topo.is_root:
        gefs.create_container(cache_dir="/pscratch/sd/t/timothys/gefs/container-cache", mode="w")
    topo.barrier()


    logger = logging.getLogger("ufs2arco")
    n_batches = len(loader)
    for batch_idx, xds in enumerate(loader):

        # xds is None if MPI rank looks for non existent indices (i.e., last batch scenario)
        # len(xds) == 0 if we couldn't find the file we were looking for
        has_content = xds is not None and len(xds) > 0
        if has_content:

            region = gefs.find_my_region(xds)
            xds.to_zarr(gefs.store_path, region=region)
            loader.clear_cache(batch_idx)


        logger.info(f"Done with batch {batch_idx+1} / {n_batches}")
