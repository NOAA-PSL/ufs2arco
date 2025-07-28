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

    Note:
        For now, all "sources" must come from the same dataset, e.g. we can't mix and match GFS and ERA5.
        There are several reasons for this:
            * Right now, this is baked into how the caching is done. We only clear one data mover's cache, and all movers use the same cache directory.
            * It's not clear how different source types (e.g. reanalysis and forecast, or deterministic and ensemble) would combine, given the different coordinates. This would make combining the datasets, and finding the specific spot in the zarr dataset (Mover.find_my_region) much more complicated.


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
        "transforms",
        "target",
        "attrs",
    )

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

    @property
    def source(self):
        return self.sources[0]

    @property
    def target(self):
        return self.targets[0]

    @property
    def mover(self):
        return self.movers[0]


    def _init_source(self):
        """Check and Initialize each Source Dataset"""

        # First, make sure the names are legit
        SourceDatasets = list()
        for local_config in self.config["multisource"]:

            name = local_config["source"]["name"].lower()
            if name not in ufs2arco.sources._recognized:
                raise NotImplementedError(f"Driver.__init__: unrecognized data source {name}. Must be one of {ufs2arco.sources._recognized}")

            SourceDatasets.append( getattr(ufs2arco.sources, ufs2arco.sources._recognized[name]) )

        try:
            assert all(local_config["source"]["name"].lower() == name for local_config in self.config["multisource"])
        except:
            raise NotImplementedError("For multisource workflows, all sources have to come from the same dataset. Can't mix and match")

        # Now initialize the actual objects
        self.sources = list()
        for LocalSource, local_kwargs in zip(SourceDatasets, self.source_kwargs):
            self.sources.append( LocalSource(**local_kwargs) )


    def _init_transformer(self):
        """Create the transformer

        For the multi source, we need a transformer for each object.
        """

        self.transformers = list()
        common_transforms = self.config.get("transforms", {})
        for local_config in self.config["multisource"]:
            unique_transforms = local_config.get("transforms", {})
            all_transforms = {**unique_transforms.copy(), **common_transforms.copy()}
            if len(all_transforms) > 0:
                transformer = Transformer(options=all_transforms)
            else:
                transformer = None
            self.transformers.append(transformer)


    def _init_target(self):
        """Make a copy of the same target format for each source

        Note that we only compute any forcings with the first data source
        """

        name = self.config["target"].get("name", "base").lower()
        try:
            assert name == "anemoi"
        except:
            raise NotImplementedError("Driver._init_target: multisource workflows currently only work with Anemoi Targets.")
        self.targets = list()
        kwargs = self.target_kwargs.copy()
        for source in self.sources:
            self.targets.append(
                ufs2arco.targets.Anemoi(
                    source=source,
                    **kwargs,
                )
            )
            # After the first source, drop "forcings" from target kwargs... no need to compute them more than once
            kwargs.pop("forcings", None)


    def _init_mover(self):

        kwargs = self.mover_kwargs.copy()
        if self.use_mpi:
            Mover = MPIDataMover
            kwargs["mpi_topo"] = self.topo
        else:
            Mover = DataMover

        self.movers = [
            Mover(source=source, target=target, transformer=transformer, **kwargs)
            for source, target, transformer in zip(self.sources, self.targets, self.transformers)
        ]


    def write_container(self, overwrite):
        """Write empty zarr store, to be filled with data"""

        if self.topo.is_root:
            dslist = [mover.create_container() for mover in self.movers]
            cds = self.target.merge_multisource(dslist)

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

            dslist = list()
            foundit = list()
            for mover in self.movers:

                xds = next(mover)

                # xds is None if MPI rank looks for non existent indices (i.e., last batch scenario)
                # len(xds) == 0 if we couldn't find the file we were looking for
                has_content = xds is not None and len(xds) > 0
                if has_content:
                    foundit.append(True)
                    dslist.append(xds.reset_coords(drop=True))

                elif xds is not None:

                    foundit.append(False)
                    batch_indices = mover.get_batch_indices(batch_idx)
                    for these_dims in batch_indices:
                        missing_dims.append(these_dims)

            # we need both conditionals, since all([]) == True
            if all(foundit) and len(foundit) == len(self.movers):

                mds = self.target.merge_multisource(dslist)

                region = self.mover.find_my_region(mds)
                mds.to_zarr(self.target.store_path, region=region)

            self.mover.clear_cache(batch_idx)

            logger.info(f"Done with batch {batch_idx+1} / {n_batches}")

        self.topo.barrier()
        logger.info(f"Done moving the data\n")

        self.report_missing_data(missing_dims)
        self.target.finalize(self.topo)
        self.finalize_attributes()

    def patch(self):
        raise NotImplementedError
