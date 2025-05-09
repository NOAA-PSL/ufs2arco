import logging
from typing import Optional

import xarray as xr

from ufs2arco.sources import Source
from ufs2arco.targets import forcings as fmod # funky name because 'forcings' seems like the more natural user specified option

logger = logging.getLogger("ufs2arco")

class Target:
    """
    Base class for the target dataset to be stored.

    This is used in the case that the target data looks "the same" as the input (i.e., the zarr form equivalent).
    """

    always_open_static_vars = False

    @property
    def sample_dims(self) -> tuple:
        return self.source.sample_dims

    @property
    def horizontal_dims(self) -> tuple:
        return self.source.horizontal_dims

    @property
    def time(self):
        if self._has_fhr:
            return None
        else:
            return self.source.time

    @property
    def t0(self):
        if self._has_fhr:
            return self.source.t0
        else:
            return None

    @property
    def fhr(self):
        if self._has_fhr:
            return self.source.fhr
        else:
            return None

    @property
    def member(self):
        if self._has_member:
            return self.source.member
        else:
            return None

    def __init__(
        self,
        source: Source,
        chunks: dict,
        store_path: str,
        rename: Optional[dict] = None,
        forcings: Optional[list | tuple] = None,
        compute_temporal_residual_statistics: Optional[bool] = False,
    ) -> None:
        """
        Initialize the GEFSDataset object.

        Args:
            source (Source): object specifying data source
            chunks (dict): Dictionary with chunk sizes for Dask arrays.
            store_path (str): Path to store the output data.
            rename (dict, optional): rename variables
            compute_temporal_residual_statistics: (bool, optional): if True, compute statistics of the temporal difference
        """
        self.source = source
        self.store_path = store_path
        self.chunks = chunks
        self.rename = rename if rename is not None else dict()

        # set these for different source handling
        self._has_fhr = getattr(self.source, "fhr", None) is not None
        self._has_member = getattr(self.source, "member", None) is not None

        # check chunks
        for dim in self.renamed_sample_dims:
            chunksize = self.chunks[dim]
            assert chunksize == 1, \
                f"{self.name}.__init__: chunks['{dim}'] = {chunksize}, but should be 1"

        # check for forcings
        recognized = tuple(fmod.get_mappings().keys())
        if forcings is not None:
            unrecognized = []
            for key in forcings:
                if key not in recognized:
                    unrecognized.append(key)
            if len(unrecognized) > 0:
                raise NotImplementedError(f"{self.name}.__init__: requested forcing variable(s) {unrecognized} are not implemented. Implemented options are {recognized}")
            self.forcings = forcings
        else:
            self.forcings = tuple()

        # temporal residual statistics
        # For now, only implemented in anemoi target
        if "anemoi" not in self.name.lower() and compute_temporal_residual_statistics:
            raise NotImplementedError(f"{self.name}.__init__: computation of temporal residual statistics not implemented for this target")
        self.compute_temporal_residual_statistics = compute_temporal_residual_statistics

        logger.info(str(self))


    def __str__(self) -> str:
        """
        Return a string representation of the GEFSDataset object.

        Returns:
            str: The string representation of the dataset.
        """
        title = f"Target: {self.name}"
        msg = f"\n{title}\n" + \
              "".join(["-" for _ in range(len(title))]) + "\n"
        for key in ["store_path", "forcings"]:
            msg += f"{key:<18s}: {getattr(self, key)}\n"
        chunkstr = "\n    ".join([f"{key:<14s}: {val}" for key, val in self.chunks.items()])
        msg += f"chunks\n    {chunkstr}\n"
        renamestr = "\n    ".join([f"{key:<14s}: {val}" for key, val in self.rename.items()])
        msg += f"rename\n    {renamestr}\n"
        return msg

    @property
    def name(self) -> str:
        return self.__class__.__name__


    @property
    def renamed_sample_dims(self) -> tuple:
        return tuple(self.rename.get(d, d) for d in self.sample_dims)


    def apply_transforms_to_sample(
        self,
        xds: xr.Dataset,
    ) -> xr.Dataset:
        """
        After opening a single dataset for the given initial condition, forecast hour, and member,
        Apply any transformations necessary for storage

        Args:
            xds (xr.Dataset): The sample dataset

        Returns:
            xr.Dataset: The dataset after any transformations
        """
        xds = self.compute_forcings(xds)
        xds = self.rename_dataset(xds)
        return xds


    def rename_dataset(self, xds: xr.Dataset) -> xr.Dataset:
        """
        This takes the default source variables and renames them to user specified options:

        Args:
            xds (xr.Dataset): a dataset directly from the source, after any selections

        Returns:
            xds (xr.Dataset): with renaming as above
        """
        for key, val in self.rename.items():
            if key in xds:
                xds = xds.rename({key: val})
        for key in self.forcings:
            dummy_name = f"computed_forcing_{key}"
            assert key not in xds, \
                f"{self.name}.rename_dataset: {key} already in dataset, remove this from forcings list"
            xds = xds.rename({dummy_name: key})
        return xds


    def compute_forcings(self, xds: xr.Dataset) -> xr.Dataset:

        time = "t0" if self._has_fhr else "time"
        mappings = fmod.get_mappings(time=time)
        for key in self.forcings:

            # make sure this dummy name is not in the dataset
            # seems ridiculous but ... who knows
            dummy_name = f"computed_forcing_{key}"
            assert dummy_name not in xds, \
                f"{self.name}.compute_forcings: {dummy_name} in dataset, but this name is needed to store forcings ... "
            func = mappings[key]
            xds[dummy_name] = func(xds)
        return xds


    def manage_coords(self, xds: xr.Dataset) -> xr.Dataset:
        """Manage the coordinates that will get stored in the container

        In the base (passive) case, create coords via the source dataset

        Args:
            xds (xr.Dataset): with all dimensions as they will be in the final dataset

        Returns:
            xds (xr.Dataset): with added / managed coordinates
        """
        xds = self.source.add_full_extra_coords(xds)
        return xds

    def aggregate_stats(self, topo) -> None:
        """Aggregate statistics over "time" and "ensemble" dimension...
        This should read in the zarr store, aggregate stats, and delete any temporary arrays from the zarr store
        """
        pass

    def calc_temporal_residual_stats(self, topo) -> None:
        pass
