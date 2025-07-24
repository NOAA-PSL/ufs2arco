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
from ufs2arco import targets
from ufs2arco.datamover import DataMover, MPIDataMover

from ufs2arco.driver import Driver

logger = logging.getLogger("ufs2arco")

class MultiDriver(Driver):
    """A class to manage data movement, with multiple sources.

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
        "multisource",
        "target",
    )

    recognized_sections = (
        "mover",
        "directories",
        "multisource",
        # TODO: common transforms
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

        # Unique Sources and Transforms
        # TODO: Note that this is redone down below for now
        self.SourceDatasets = list()
        self.UniqueTransforms = list()
        for local_config in self.config["multisource"]:

            # First the source
            name = local_config["source"]["name"].lower()
            if name not in ufs2arco.sources._recognized:
                raise NotImplementedError(f"Driver.__init__: unrecognized data source {name}. Must be one of {ufs2arco.sources._recognized}")

            self.SourceDatasets.append( getattr(ufs2arco.sources, ufs2arco.sources._recognized[name]) )

        try:
            assert all(local_config["source"]["name"].lower() == name for local_config in self.config["multisource"])
        except:
            raise NotImplementedError("For now, I'm assuming all sources are the same, for convenient caching reasons")

        # the target
        name = self.config["target"].get("name", "base")
        name = name.lower()
        if name in ("forecast", "analysis", "base"):
            self.TargetDataset = targets.Target
            raise NotImplementedError("Driver.__init__: it is unclear how multisource will work with base Target")

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
    def source_kwargs(self) -> dict:
        """Returns the arguments for initializing the source dataset.

        Returns:
            dict: The source dataset initialization arguments.
        """
        return [
            {key: val for key, val in local_config["source"].items() if key != "name"}
            for local_config in self.config["multisource"]
        ]


    def setup(self, runtype):
        # MPI requires some extra setup
        mover_kwargs = self.mover_kwargs.copy()

        log_dir = os.path.expandvars(self.config["directories"]["logs"])

        if runtype != "create":
            raise NotImplementedError

        if self.use_mpi:
            topo = MPITopology(log_dir=log_dir)
            mover_kwargs["mpi_topo"] = topo

        else:
            topo = SerialTopology(log_dir=log_dir)

        sources = list()
        unique_transformers = list()

        for LocalSource, local_kwargs in zip(self.SourceDatasets, self.source_kwargs):
            sources.append( LocalSource(**local_kwargs) )

        sources = list()
        unique_transformers = list()
        # TODO: common_transformers = list()
        # This is made complicated by the fact that the mover takes one transformer currently
        targets = list()
        movers = list()
        target_kwargs = self.target_kwargs.copy()
        for local_config in self.config["multisource"]:

            # Each Source
            name = local_config["source"]["name"]
            SourceDataset = getattr(ufs2arco.sources, ufs2arco.sources._recognized[name])
            source_kw = {key: val for key, val in local_config["source"].items() if key != "name"}
            source = SourceDataset(**source_kw)
            sources.append(source)

            # If Unique transforms are desired, get them here
            if "transforms" in local_config.keys():
                transformer = Transformer(options=local_config["transforms"])
            else:
                transformer = None
            unique_transformers.append(transformer)

            # Create a separate target for each, using the same definition
            target = self.TargetDataset(source=source, **target_kwargs)
            targets.append(target)

            # And mover
            mover = self.Mover(source=source, target=target, transformer=transformer, **mover_kwargs)
            movers.append(mover)

            # After the first source, drop "forcings" from target kwargs... no need to compute them more than once
            target_kwargs.pop("forcings", None)
        return topo, movers, sources, targets

    def write_container(self, movers, targets, overwrite):
        """Here we need a custom container creation to handle multiple sources...
        Basically the lines below to handle a single batch, then with the mover container creation, with the addition of targets[0].merge_multisource
        """

        if movers[0].topo.is_root:
            dslist = [mover.create_container() for mover in movers]
            cds = targets[0].merge_multisource(dslist)

            container_kwargs = {"mode": "w"} if overwrite else {}
            logger.info(f"Driver.write_container: storing container at {targets[0].store_path}\n{cds}\n")
            cds.to_zarr(targets[0].store_path, compute=False, **container_kwargs)
            logger.info("Driver.write_container: Done storing container.\n")

        movers[0].topo.barrier()


    def run(self, overwrite: bool = False):
        """Runs the data movement process, managing the datasets and mover.

        This method sets up the datasets, creates the container, and loops through
        batches to move data to the specified store path (Zarr format).

        Args:
            overwrite (bool, optional): Whether to overwrite the existing container.
                Defaults to False.
        """
        topo, movers, sources, targets = self.setup(runtype="create")

        # create container, only if mover start is not 0
        if movers[0].start == 0:
            self.write_container(movers, targets, overwrite=overwrite)

        # loop through batches
        n_batches = len(movers[0])
        missing_dims = []
        for batch_idx in range(movers[0].start, n_batches):

            dslist = list()
            for mover in movers:

                xds = next(mover)

                # xds is None if MPI rank looks for non existent indices (i.e., last batch scenario)
                # len(xds) == 0 if we couldn't find the file we were looking for
                has_content = xds is not None and len(xds) > 0
                if has_content:
                    dslist.append(xds.reset_coords(drop=True))

                elif xds is not None:
                    # exit, no need to check the other sources
                    # we don't continue with partial data
                    batch_indices = mover.get_batch_indices(batch_idx)
                    for these_dims in batch_indices:
                        missing_dims.append(these_dims)
                    break

            if len(dslist) == len(movers):

                xds = targets[0].merge_multisource(dslist)

                # Note that the find_my_region clearly works using any of the movers
                # for an anemoi target, and with deterministic data
                # With ensemble data, all sources need the same number of ensemble members
                # and this becomes way less clear with the base style
                region = movers[0].find_my_region(xds) # k
                xds.to_zarr(targets[0].store_path, region=region)
                # TODO: do we always cache all potential files? if so, we only need 1 cache for all movers
                # ... assuming they're from the same source!
                movers[0].clear_cache(batch_idx)

            logger.info(f"Done with batch {batch_idx+1} / {n_batches}")

        topo.barrier()
        logger.info(f"Done moving the data\n")

        self.report_missing_data(topo, sources[0], targets[0], missing_dims)
        targets[0].finalize(topo)
        self.finalize_attributes(topo, targets[0])

    def patch(self):
        raise NotImplemented

    def finalize_attributes(self, topo, target):

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


    def get_missing_data_path(self, store_path) -> str:
        directory, zstore = os.path.split(store_path)
        return f"{directory}/missing.{zstore}.yaml"

    def report_missing_data(self, topo, source, target, missing_dims):

        missing_dims = topo.gather(missing_dims)

        if topo.is_root:
            # collect the list of lists into one list of dicts
            missing_dims = [item for sublist in missing_dims for item in sublist]

            if len(missing_dims) > 0:
                # sort it by t0 or time
                # then convert numpy or pandas types to int/str etc
                missing_dims = sorted(missing_dims, key=_get_time)
                missing_dims = [_convert_types_to_yaml(d) for d in missing_dims]

                target.handle_missing_data(missing_dims)

                missing_data_yaml = self.get_missing_data_path(target.store_path)
                with open(missing_data_yaml, "w") as f:
                    yaml.dump(missing_dims, stream=f)

                logger.warning(f"⚠️  Some data are missing.")
                logger.warning(f"⚠️  The missing dimension combos, i.e., {source.sample_dims}")
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
