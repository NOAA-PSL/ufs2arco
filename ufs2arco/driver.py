try:
    from mpi4py import MPI
except:
    pass

import os
import logging
import yaml
from datetime import datetime

import numpy as np
import xarray as xr
import pandas as pd
import zarr

from ufs2arco.log import setup_simple_log
from ufs2arco.mpi import MPITopology, SerialTopology
import ufs2arco.sources
from ufs2arco.transforms import Transformer
import ufs2arco.targets
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

        Note:
            Initialization does very little, since we need to start the run in order to setup the MPI or Serial Topology that orchestrates everything. This is used for logging. After that, we can initialize the source, target, etc.

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

        # directories
        dirs = self.config["directories"]
        for key in ["zarr", "cache"]:
            assert key in dirs, \
                f"Driver.__init__: could not find '{key}' in 'directories' section in yaml"

        for key in ["cache_dir"]:
            assert key not in self.config["mover"], \
                f"Driver.__init__: '{key}' not allowed in 'mover' section of yaml"

        # make sure chunks are here
        for key in ["chunks"]:
            assert key in self.config["target"], \
                f"Driver.__init__: could not find '{key}' in 'target' section of yaml"


    def _init_topo(self, runtype: str):
        """Initialize MPITopology or SerialTopology, which sets up the logging"""

        mover_kwargs = self.mover_kwargs.copy()

        log_dir = os.path.expandvars(self.config["directories"]["logs"])

        if runtype != "create":
            if log_dir.endswith("/"):
                log_dir = log_dir[:-1]
            log_dir += f"-{runtype}"

        if self.use_mpi:
            self.topo = MPITopology(log_dir=log_dir)

        else:
            self.topo = SerialTopology(log_dir=log_dir)


    def _init_source(self):
        """Check and Initialize the Source Dataset and Transformer"""

        name = self.config["source"]["name"].lower()
        if name not in ufs2arco.sources._recognized:
            raise NotImplementedError(f"Driver._init_source_and_transforms: unrecognized data source {name}. Must be one of {ufs2arco.sources._recognized}")
        SourceDataset = getattr(ufs2arco.sources, ufs2arco.sources._recognized[name])

        self.source = SourceDataset(**self.source_kwargs)

    def _init_transformer(self):
        """create the transformer"""

        if "transforms" in self.config.keys():
            self.transformer = Transformer(options=self.config["transforms"])
        else:
            self.transformer = None


    def _init_target(self):
        """Check and Initialize the Target Dataset format"""

        name = self.config["target"].get("name", "base").lower()
        if name in ("forecast", "analysis", "base"):
            TargetDataset = ufs2arco.targets.Target
        elif name == "anemoi":
            TargetDataset = ufs2arco.targets.Anemoi
        else:
            raise NotImplementedError(f"Driver._init_targets: only 'base' and 'anemoi' are implemented")

        self.target = TargetDataset(source=self.source, **self.target_kwargs)


    def _init_mover(self):
        """Initialize the DataMover"""

        kwargs = self.mover_kwargs.copy()
        if self.use_mpi:
            Mover = MPIDataMover
            kwargs["mpi_topo"] = self.topo
        else:
            Mover = DataMover

        self.mover = Mover(source=self.source, target=self.target, transformer=self.transformer, **kwargs)


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


    @property
    def store_path(self) -> str:
        return self.target.store_path


    def setup(self, runtype: str):
        self._init_topo(runtype)
        self._init_source()
        self._init_transformer()
        self._init_target()
        self._init_mover()


    def write_container(self, overwrite):
        """Write empty zarr store, to be filled with data"""

        if self.topo.is_root:
            cds = self.mover.create_container()

            kwargs = {"mode": "w"} if overwrite else {}
            logger.info(f"Driver.write_container: storing container at {self.store_path}\n{cds}\n")
            cds.to_zarr(self.store_path, compute=False, **kwargs)
            logger.info("Driver.write_container: Done storing container.\n")

        self.topo.barrier()


    def run(self, overwrite: bool = False):
        """Runs the data movement process, managing the datasets and mover.

        This method sets up the datasets, creates the container, and loops through
        batches to move data to the specified store path (Zarr format).

        Args:
            overwrite (bool, optional): Whether to overwrite the existing container.
                Defaults to False.
        """
        self.setup(runtype="create")

        # create container, only if mover start is not 0
        if self.mover.start == 0:
            self.write_container(overwrite=overwrite)

        # loop through batches
        n_batches = len(self.mover)
        missing_dims = []
        for batch_idx in range(self.mover.start, n_batches):

            xds = next(self.mover)


            # xds is None if MPI rank looks for non existent indices (i.e., last batch scenario)
            # len(xds) == 0 if we couldn't find the file we were looking for
            has_content = xds is not None and len(xds) > 0
            if has_content:

                xds = xds.reset_coords(drop=True)
                region = self.mover.find_my_region(xds)
                xds.to_zarr(self.store_path, region=region)
                self.mover.clear_cache(batch_idx)

            elif xds is not None:

                # we couldn't find the file, keep track of it
                batch_indices = self.mover.get_batch_indices(batch_idx)
                for these_dims in batch_indices:
                    missing_dims.append(these_dims)

            logger.info(f"Done with batch {batch_idx+1} / {n_batches}")

        self.topo.barrier()
        logger.info(f"Done moving the data\n")

        self.report_missing_data(missing_dims)
        self.target.finalize(self.topo)
        self.finalize_attributes()

    def patch(self):

        self.setup(runtype="patch")
        missing_dims = _open_patch_yaml(self.get_missing_data_path(self.store_path))

        logger.info(f"Starting patch workflow with missing_dims\n{missing_dims}\n")

        # hacky for now, reset the mover's sample_indices to be the missing dims
        self.mover.sample_indices = missing_dims
        self.mover.restart(idx=0)

        missing_again = list()
        n_batches = len(self.mover)
        for batch_idx, xds in enumerate(self.mover):

            # xds is None if MPI rank looks for non existent indices (i.e., last batch scenario)
            # len(xds) == 0 if we couldn't find the file we were looking for
            has_content = xds is not None and len(xds) > 0
            if has_content:

                xds = xds.reset_coords(drop=True)
                region = self.mover.find_my_region(xds)
                xds.to_zarr(self.store_path, region=region)
                self.mover.clear_cache(batch_idx)

            elif xds is not None:

                batch_indices = self.mover.get_batch_indices(batch_idx)
                for these_dims in batch_indices:
                    missing_again.append(these_dims)

            logger.info(f"Done with batch {batch_idx+1} / {n_batches}")

        self.topo.barrier()
        logger.info(f"Done moving the data\n")

        self.report_missing_data(missing_again)
        self.target.finalize(self.topo)
        self.finalize_attributes()


    def finalize_attributes(self):

        logger.info(f"Storing the recipe and anything from the 'attrs' section in zarr store attributes")
        if self.topo.is_root:
            zds = zarr.open(self.store_path, mode="a")
            zds.attrs["recipe"] = self.config
            if "attrs" in self.config.keys():
                for key, val in self.config["attrs"].items():
                    zds.attrs[key] = val

            zds.attrs["latest_write_timestamp"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            # just in case
            zarr.consolidate_metadata(self.store_path)
        self.topo.barrier()
        logger.info(f"🚀🚀🚀 Dataset is ready for launch at: {self.store_path}")


    def get_missing_data_path(self, store_path) -> str:
        directory, zstore = os.path.split(store_path)
        return f"{directory}/missing.{zstore}.yaml"

    def report_missing_data(self, missing_dims):

        missing_dims = self.topo.gather(missing_dims)

        if self.topo.is_root:
            # collect the list of lists into one list of dicts
            missing_dims = [item for sublist in missing_dims for item in sublist]

            if len(missing_dims) > 0:
                # sort it by t0 or time
                # then convert numpy or pandas types to int/str etc
                missing_dims = sorted(missing_dims, key=_get_time)
                missing_dims = [_convert_types_to_yaml(d) for d in missing_dims]

                self.target.handle_missing_data(missing_dims)

                missing_data_yaml = self.get_missing_data_path(self.store_path)
                with open(missing_data_yaml, "w") as f:
                    yaml.dump(missing_dims, stream=f)

                logger.warning(f"⚠️  Some data are missing.")
                logger.warning(f"⚠️  The missing dimension combos, i.e., {self.source.sample_dims}")
                logger.warning(f"⚠️  were written to: {missing_data_yaml}")
                logger.warning(f"If you know the files are actually available, You can run")
                logger.warning(f"\tpython -c 'import ufs2arco; ufs2arco.Driver(\"/path/to/your/original/recipe.yaml\").patch()'")
                logger.warning(f"to try getting the data again\n")

# some utilities for handling missing data
def _get_time(d):
    return d.get("t0", d.get("time", None))

def _convert_types_to_yaml(d):
    d = d.copy()
    # Convert pd.Timestamp to string
    if "t0" in d and isinstance(d["t0"], pd.Timestamp):
        d["t0"] = str(d["t0"])
    if "time" in d and isinstance(d["time"], pd.Timestamp):
        d["time"] = str(d["time"])
    # Convert numpy integers to Python int
    for key in ["fhr", "member"]:
        if key in d and isinstance(d[key], np.integer):
            d[key] = int(d[key])
    return d

def _open_patch_yaml(yamlpath):

    with open(yamlpath, "r") as f:
        missing_dims = yaml.safe_load(f)
    # Convert string -> pd.Timestamp
    for i, d in enumerate(missing_dims):
        if "t0" in d and isinstance(d["t0"], str):
            missing_dims[i]["t0"] = pd.Timestamp(d["t0"])
        if "time" in d and isinstance(d["time"], str):
            missing_dims[i]["time"] = pd.Timestamp(d["time"])
    return missing_dims
