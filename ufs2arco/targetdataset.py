import logging
from typing import Optional

import xarray as xr

from ufs2arco.sourcedataset import SourceDataset

logger = logging.getLogger("ufs2arco")

class TargetDataset:
    """Base class for the target dataset to be stored
    """

    sample_dims = tuple()
    base_dims = tuple()
    always_open_static_vars = False

    def __init__(
        self,
        source: SourceDataset,
        chunks: dict,
        store_path: str,
        rename: Optional[dict] = None,
    ) -> None:
        """
        Initialize the GEFSDataset object.

        Args:
            chunks (dict): Dictionary with chunk sizes for Dask arrays.
            store_path (str): Path to store the output data.
        """
        self.source = source
        self.store_path = store_path
        self.chunks = chunks
        self.rename = rename if rename is not None else dict()
        for dim in self.renamed_sample_dims:
            chunksize = self.chunks[dim]
            assert chunksize == 1, \
                f"{self.name}.__init__: chunks['{dim}'] = {chunksize}, but should be 1"
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
        for key in ["store_path"]:
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
        pass

    def manage_coords(self, xds: xr.Dataset) -> xr.Dataset:
        """Manage the coordinates that will get stored in the container

        Args:
            xds (xr.Dataset): with all dimensions as they will be in the final dataset

        Returns:
            xds (xr.Dataset): with added / managed coordinates
        """
        pass
