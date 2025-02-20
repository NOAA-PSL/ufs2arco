try:
    from mpi4py import MPI
except:
    pass

import logging
import yaml

from ufs2arco.mpi import MPITopology
from ufs2arco.gefsdataset import GEFSDataset
from ufs2arco.datamover import DataMover, MPIDataMover

class Driver:
    """A class to manage data movement.

    Attributes:
        config (dict): Configuration dictionary loaded from the YAML file.
        Dataset (Type[GEFSDataset]): The dataset class (GEFSDataset).
        Mover (Type[DataMover] | Type[MPIDataMover]): The data mover class (DataMover or MPIDataMover).

    Methods:
        __init__(config_filename: str): Initializes the Driver object with configuration from the specified YAML file.
        use_mpi: Returns whether MPI should be used based on the configuration.
        dataset_kwargs: Returns the arguments for initializing the dataset.
        mover_kwargs: Returns the arguments for initializing the mover.
        run(overwrite: bool = False): Runs the data movement process, managing the dataset and mover.
    """

    def __init__(self, config_filename: str):
        """Initializes the Driver object with configuration from the specified YAML file.

        Args:
            config_filename (str): Path to the YAML configuration file.

        Raises:
            AssertionError: If required sections or keys are missing in the configuration.
            NotImplementedError: If a dataset or mover is not recognized.
        """
        with open(config_filename, "r") as f:
            self.config = yaml.safe_load(f)

        for key in ["dataset", "mover", "directories"]:
            assert key in self.config, \
                f"Driver.__init__: could not find '{key}' section in yaml"

        # the dataset
        name = self.config["dataset"]["name"].lower()
        if name == "gefsdataset":
            self.Dataset = GEFSDataset
        else:
            raise NotImplementedError(f"Driver.__init__: only 'GEFSDataset' is implemented")

        # the mover
        for key in ["name", "sample_dims"]:
            assert key in self.config["mover"], \
                f"Driver.__init__: could not find '{key}' in 'mover' section in yaml"
        for key in ["cache_dir", "num_workers", "max_queue_size"]:
            assert key not in self.config["mover"], \
                f"Driver.__init__: '{key}' not allowed in 'mover' section in yaml"

        name = self.config["mover"]["name"].lower()
        if name == "datamover":
            self.Mover = DataMover
            logger = logging.getLogger("ufs2arco")
            logger.warning("Driver.__init__: unsure what will happen with logs directory using non MPIDataMover")
        elif name == "mpidatamover":
            self.Mover = MPIDataMover
        else:
            raise NotImplementedError(f"Driver.__init__: don't recognize mover = {name}")

        # directories
        dirs = self.config["directories"]
        for key in ["zarr", "cache", "logs"]:
            assert key in dirs, \
                f"Driver.__init__: could not find '{key}' in 'directories' section in yaml"

    @property
    def use_mpi(self) -> bool:
        """Determines if MPI is to be used based on the mover configuration.

        Returns:
            bool: True if MPI is to be used, False otherwise.
        """
        return self.config["mover"]["name"].lower() == "mpidatamover"

    @property
    def dataset_kwargs(self) -> dict:
        """Returns the arguments for initializing the dataset.

        Returns:
            dict: The dataset initialization arguments.
        """
        kw = {key: val for key, val in self.config["dataset"].items() if key != "name"}
        kw["store_path"] = self.config["directories"]["zarr"]
        return kw

    @property
    def mover_kwargs(self) -> dict:
        """Returns the arguments for initializing the data mover.

        Returns:
            dict: The mover initialization arguments.
        """
        kw = {key: val for key, val in self.config["mover"].items() if key != "name"}

        # optional
        if "start" not in kw.keys():
            kw["start"] = 0

        # enforced options
        kw["num_workers"] = 0
        kw["max_queue_size"] = 1
        kw["cache_dir"] = self.config["directories"]["cache"]
        return kw

    def run(self, overwrite: bool = False):
        """Runs the data movement process, managing the dataset and mover.

        This method sets up the dataset, creates the container, and loops through
        batches to move data to the specified store path (Zarr format).

        Args:
            overwrite (bool, optional): Whether to overwrite the existing container.
                Defaults to False.
        """
        # MPI requires some extra setup
        mover_kwargs = self.mover_kwargs.copy()
        if self.use_mpi:
            topo = MPITopology(log_dir=self.config["directories"]["logs"])
            mover_kwargs["mpi_topo"] = topo

        dataset = self.Dataset(**self.dataset_kwargs)
        mover = self.Mover(dataset=dataset, **mover_kwargs)
        logger = logging.getLogger("ufs2arco")

        # create container, only if mover start is not 0
        if mover_kwargs["start"] == 0:
            container_kwargs = {"mode": "w"} if overwrite else {}
            container_cache = self.config["directories"]["cache"] + "/container"
            if self.use_mpi:
                if topo.is_root:
                    dataset.create_container(cache_dir=container_cache, **container_kwargs)
                topo.barrier()
            else:
                dataset.create_container(cache_dir=container_cache, **container_kwargs)

        # loop through batches
        n_batches = len(mover)
        for batch_idx in range(mover_kwargs["start"], len(mover)):

            xds = next(mover)

            # xds is None if MPI rank looks for non existent indices (i.e., last batch scenario)
            # len(xds) == 0 if we couldn't find the file we were looking for
            has_content = xds is not None and len(xds) > 0
            if has_content:

                xds = xds.reset_coords(drop=True)
                region = mover.find_my_region(xds)
                xds.to_zarr(dataset.store_path, region=region)
                mover.clear_cache(batch_idx)

            logger.info(f"Done with batch {batch_idx+1} / {n_batches}")
