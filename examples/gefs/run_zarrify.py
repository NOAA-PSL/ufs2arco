import logging
import pandas as pd

from ufs2arco.mpi import MPITopology
from ufs2arco.gefsdataset import GEFSDataset
from ufs2arco.batchloader import BatchLoader

if __name__ == "__main__":

    config = "./recipe.yaml" # is this even necessary if we could put it all in this script?
    gefs = GEFSDataset(config)

    #topo = MPITopology(log_dir="./logs")

    loader = BatchLoader(
        dataset=gefs,
        batch_size=4,
        num_workers=0,
        max_queue_size=1,
    #    mpi_topo=topo,
    )

    # which requires a store location...
    gefs.create_container(mode="w")

    n_batches = len(loader)
    for batch_idx, xds in enumerate(loader):
        if xds is not None:

            # figure out what dates we're working with, and the indices relative to the global list
            batch_dates = [pd.Timestamp(t0) for t0 in xds["t0"].values]
            date_indices = [list(gefs.dates).index(date) for date in batch_dates]

            region = {k: slice(None, None) for k in xds.dims}
            region["t0"] = slice(date_indices[0], date_indices[-1]+1)

            # TODO: need that store location
            xds.to_zarr(gefs.store_path, region=region)
            loader.clear_cache(batch_idx)

        logging.info(f"Done with batch {batch_idx+1} / {n_batches}")
        logging.info(f"    Dates: {batch_dates[0]} - {batch_dates[-1]}")

    #    # Potential TODO:
    #    # add progress tracker
