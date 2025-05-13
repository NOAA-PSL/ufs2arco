try:
    from mpi4py import MPI
except:
    pass

import os
import logging
import yaml
from datetime import datetime

import xarray as xr
import zarr

from ufs2arco.log import setup_simple_log
from ufs2arco.mpi import MPITopology, SerialTopology
from ufs2arco import sources
from ufs2arco.transforms import Transformer
from ufs2arco import targets
from ufs2arco.datamover import DataMover, MPIDataMover

logger = logging.getLogger("ufs2arco")

class Driver:
    """A class to manage data movement.

    Attributes:
        config (dict): Configuration dictionary loaded from the YAML file.
        Source, Transformer, Target
        Mover (Type[DataMover] | Type[MPIDataMover]): The data mover class (DataMover or MPIDataMover).

    Methods:
        __init__(config_filename: str): Initializes the Driver object with configuration from the specified YAML file.
        use_mpi: Returns whether MPI should be used based on the configuration.
        source_kwargs: Returns the arguments for initializing the source dataset.
        target_kwargs: Returns the arguments for initializing the target dataset.
        mover_kwargs: Returns the arguments for initializing the mover.
        run(overwrite: bool = False): Runs the data movement process, managing the source, target transformations, and mover.
    """

    required_sections = (
        "mover",
        "directories",
        "source",
        "target",
    )

    recognized_sections = (
        "mover",
        "directories",
        "source",
        "transforms",
        "target",
        "attrs",
    )

    def __init__(self, config_filename: str):
        """Initializes the Driver object with configuration from the specified YAML file.

        Args:
            config_filename (str): Path to the YAML configuration file.

        Raises:
            AssertionError: If required sections or keys are missing in the configuration.
            NotImplementedError: If a source, target, or mover is not recognized.
        """
        with open(config_filename, "r") as f:
            self.config = yaml.safe_load(f)

        for key in self.required_sections:
            assert key in self.config, \
                f"Driver.__init__: could not find '{key}' section in yaml"

        unrecognized = []
        for key in self.config.keys():
            if key not in self.recognized_sections:
                unrecognized.append(key)
        if len(unrecognized) > 0:
            raise KeyError(
                f"Driver.__init__: Unrecognized config sections: {unrecognized}. The following are recognized: {self.recognized_sections}")

        # the source dataset
        name = self.config["source"]["name"].lower()
        if name not in sources._recognized:
            raise NotImplementedError(f"Driver.__init__: unrecognized data source {name}. Must be one of {sources._recognized}")
        self.SourceDataset = getattr(sources, sources._recognized[name])

        # the target
        name = self.config["target"].get("name", "base")
        name = name.lower()
        if name in ("forecast", "analysis", "base"):
            self.TargetDataset = targets.Target
        elif name == "anemoi":
            self.TargetDataset = targets.Anemoi
        else:
            raise NotImplementedError(f"Driver.__init__: only 'base' and 'anemoi' are implemented")

        for key in ["chunks"]:
            assert key in self.config["target"], \
                f"Driver.__init__: could not find '{key}' in 'target' section of yaml"

        # the mover
        name = self.config["mover"]["name"].lower()
        if name == "datamover":
            self.Mover = DataMover
        elif name == "mpidatamover":
            self.Mover = MPIDataMover
        else:
            raise NotImplementedError(f"Driver.__init__: don't recognize mover = {name}")

        for key in ["cache_dir"]:
            assert key not in self.config["mover"], \
                f"Driver.__init__: '{key}' not allowed in 'mover' section of yaml"

        # directories
        dirs = self.config["directories"]
        for key in ["zarr", "cache"]:
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
    def source_kwargs(self) -> dict:
        """Returns the arguments for initializing the source dataset.

        Returns:
            dict: The source dataset initialization arguments.
        """
        kw = {key: val for key, val in self.config["source"].items() if key != "name"}
        return kw

    @property
    def target_kwargs(self) -> dict:
        """Returns the arguments for initializing the target dataset.

        Returns:
            dict: The target dataset initialization arguments.
        """
        kw = {key: val for key, val in self.config["target"].items() if key != "name"}
        kw["store_path"] = os.path.expandvars(self.config["directories"]["zarr"])
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
        kw["cache_dir"] = os.path.expandvars(self.config["directories"]["cache"])
        return kw

    def run(self, overwrite: bool = False):
        """Runs the data movement process, managing the datasets and mover.

        This method sets up the datasets, creates the container, and loops through
        batches to move data to the specified store path (Zarr format).

        Args:
            overwrite (bool, optional): Whether to overwrite the existing container.
                Defaults to False.
        """
        # MPI requires some extra setup
        mover_kwargs = self.mover_kwargs.copy()
        if self.use_mpi:
            topo = MPITopology(log_dir=os.path.expandvars(self.config["directories"]["logs"]))
            mover_kwargs["mpi_topo"] = topo

        else:
            topo = SerialTopology(log_dir=os.path.expandvars(self.config["directories"]["logs"]))

        source = self.SourceDataset(**self.source_kwargs)
        # create the transformer
        if "transforms" in self.config.keys():
            self.transformer = Transformer(options=self.config["transforms"])
        else:
            self.transformer = None
        target = self.TargetDataset(source=source, **self.target_kwargs)
        mover = self.Mover(source=source, target=target, transformer=self.transformer, **mover_kwargs)

        # create container, only if mover start is not 0
        if mover_kwargs["start"] == 0:
            container_kwargs = {"mode": "w"} if overwrite else {}
            mover.create_container(**container_kwargs)

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
                xds.to_zarr(target.store_path, region=region)
                mover.clear_cache(batch_idx)

            logger.info(f"Done with batch {batch_idx+1} / {n_batches}")

        topo.barrier()
        logger.info(f"Done moving the data\n")

        logger.info(f"Aggregating statistics (if any specified for target)")
        target.aggregate_stats(topo)
        logger.info(f"Done aggregating statistics\n")

        if target.compute_temporal_residual_statistics:
            logger.info(f"Computing temporal residual statistics")
            target.calc_temporal_residual_stats(topo)
            logger.info(f"Done computing temporal residual statistics")

        logger.info(f"Storing the recipe and anything from the 'attrs' section in zarr store attributes")
        if topo.is_root:
            zds = zarr.open(target.store_path, mode="a")
            zds.attrs["recipe"] = self.config
            if "attrs" in self.config.keys():
                for key, val in self.config["attrs"].items():
                    zds.attrs[key] = val

            zds.attrs["latest_write_timestamp"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            # just in case
            zarr.consolidate_metadata(target.store_path)
        topo.barrier()

        logger.info(f"🚀🚀🚀 Dataset is ready for launch at: {target.store_path}")
