import logging
from typing import Optional

import xarray as xr

from ufs2arco.sources import Source

logger = logging.getLogger("ufs2arco")

class Target:
    """Base class for the target dataset to be stored
    """

    sample_dims = tuple()
    base_dims = tuple()
    always_open_static_vars = False

    def __init__(
        self,
        source: Source,
        chunks: dict,
        store_path: str,
        rename: Optional[dict] = None,
        slices: Optional[dict] = None,
    ) -> None:
        """
        Initialize the GEFSDataset object.

        Args:
            source (Source): object specifying data source
            chunks (dict): Dictionary with chunk sizes for Dask arrays.
            store_path (str): Path to store the output data.
            rename (dict, optional): rename variables
            slices (dict, optional): either "sel" or "isel", with slice, passed to xarray
        """
        self.source = source
        self.store_path = store_path
        self.chunks = chunks
        self.rename = rename if rename is not None else dict()
        for dim in self.renamed_sample_dims:
            chunksize = self.chunks[dim]
            assert chunksize == 1, \
                f"{self.name}.__init__: chunks['{dim}'] = {chunksize}, but should be 1"

        # check for slicing
        recognized = ("sel", "isel")
        for key in slices.keys():
            if key not in recognized:
                raise NotImplementedError(f"{self.name}.__init__: can't use {key} slice, only {recognized} are recognized so far...")
        self.slices = slices if slices is not None else dict()

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
        for key in ["store_path", "slices"]:
            msg += f"{key:<18s}: {getattr(self, key)}\n"
        chunkstr = "\n    ".join([f"{key:<14s}: {val}" for key, val in self.chunks.items()])
        msg += f"chunks\n    {chunkstr}\n"
        return msg

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def renamed_sample_dims(self) -> tuple:
        return tuple(self.rename[d] if d in self.rename else d for d in self.sample_dims)


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
        xds = self.apply_slices(xds)
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
        return xds


    def apply_slices(self, xds: xr.Dataset) -> xr.Dataset:
        """Apply any slices, for now just data selection via "sel" or "isel"
        Note that this is the first transformation, so slicing options relate to
        the standard dimensions:

            (t0, fhr, member, level, latitude, longitude)

        Args:
            xds (xr.Dataset): The sample dataset

        Returns:
            xr.Dataset: The dataset after slices applied
        """

        if "sel" in self.slices.keys():
            for key, val in self.slices["sel"].items():
                xds = xds.sel({key: slice(*val)})

        if "isel" in self.slices.keys():
            for key, val in self.slices["isel"].items():
                xds = xds.isel({key: slice(*val)})
        return xds


    def manage_coords(self, xds: xr.Dataset) -> xr.Dataset:
        """Manage the coordinates that will get stored in the container

        Args:
            xds (xr.Dataset): with all dimensions as they will be in the final dataset

        Returns:
            xds (xr.Dataset): with added / managed coordinates
        """
        pass

    def aggregate_stats(self) -> None:
        """Aggregate statistics over "time" and "ensemble" dimension...
        This should read in the zarr store, aggregate stats, and delete any temporary arrays from the zarr store
        """
        pass
