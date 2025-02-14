try:
    from mpi4py import MPI
except:
    pass

import logging
import yaml

from ufs2arco.mpi import MPITopology
from ufs2arco.gefsdataset import GEFSDataset
from ufs2arco.batchloader import BatchLoader, MPIBatchLoader

class Driver():
    """A class to manage all of the data movement
    """

    def __init__(self, config_filename: str):

        with open(config_filename, "r") as f:
            self.config = yaml.safe_load(f)

        for key in ["dataset", "loader", "directories"]:
            assert key in self.config, \
                f"Driver.__init__: could not find '{key}' section in yaml"

        # the dataset
        name = self.config["dataset"]["name"].lower()
        if name == "gefsdataset":
            self.Dataset = GEFSDataset
        else:
            raise NotImplementedError(f"Driver.__init__: only 'GEFSDataset' is implemented")

        # the loader
        for key in ["name", "sample_dims"]:
            assert key in self.config["loader"], \
                f"Driver.__init__: could not find '{key}' in 'loader' section in yaml"
        for key in ["cache_dir", "num_workers", "max_queue_size"]:
            assert key not in self.config["loader"], \
                f"Driver.__init__: '{key}' not allowed in 'loader' section in yaml"

        name = self.config["loader"]["name"].lower()
        if name == "batchloader":
            self.Loader = BatchLoader
            logger = logging.getLogger("ufs2arco")
            logger.warning("Driver.__init__: unsure what will happen with logs directory using non MPIBatchLoader")
        elif name == "mpibatchloader":
            self.Loader = MPIBatchLoader
        else:
            raise NotImplementedError(f"Driver.__init__: don't recognize loader = {name}")

        # directories
        dirs = self.config["directories"]
        for key in ["zarr", "cache", "logs"]:
            assert key in dirs, \
                f"Driver.__init__: could not find '{key}' in 'directories' section in yaml"

    @property
    def use_mpi(self):
        return self.config["loader"]["name"].lower() == "mpibatchloader"

    @property
    def dataset_kwargs(self):
        kw = {key: val for key, val in self.config["dataset"].items() if key != "name"}
        kw["store_path"] = self.config["directories"]["zarr"]
        return kw

    @property
    def loader_kwargs(self):
        kw = {key: val for key, val in self.config["loader"].items() if key != "name"}
        # optional
        if "start" not in kw.keys():
            kw["start"] = 0
        # enforced options
        kw["num_workers"] = 0
        kw["max_queue_size"] = 1
        kw["cache_dir"] = self.config["directories"]["cache"]
        return kw

    def run(self, overwrite: bool = False):

        # MPI requires some extra setup
        loader_kwargs = self.loader_kwargs.copy()
        if self.use_mpi:
            topo = MPITopology(log_dir=self.config["directories"]["logs"])
            loader_kwargs["mpi_topo"] = topo

        dataset = self.Dataset(**self.dataset_kwargs)
        loader = self.Loader(dataset=dataset, **loader_kwargs)
        logger = logging.getLogger("ufs2arco")

        # create container, only if loader start is not 0
        if loader_kwargs["start"] == 0:
            container_kwargs = {"mode": "w"} if overwrite else {}
            container_cache = self.config["directories"]["cache"]+"/container"
            if self.use_mpi:
                if topo.is_root:
                    dataset.create_container(cache_dir=container_cache, **container_kwargs)
                topo.barrier()
            else:
                dataset.create_container(cache_dir=container_cache, **container_kwargs)

        # loop through batches
        n_batches = len(loader)
        for batch_idx, xds in enumerate(loader):

            # xds is None if MPI rank looks for non existent indices (i.e., last batch scenario)
            # len(xds) == 0 if we couldn't find the file we were looking for
            has_content = xds is not None and len(xds) > 0
            if has_content:

                xds = xds.reset_coords(drop=True)
                region = loader.find_my_region(xds)
                xds.to_zarr(dataset.store_path, region=region)
                loader.clear_cache(batch_idx)

            logger.info(f"Done with batch {batch_idx+1} / {n_batches}")
