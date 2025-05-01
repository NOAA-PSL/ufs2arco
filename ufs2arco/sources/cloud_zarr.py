import logging
from typing import Optional
import yaml

import pandas as pd
import xarray as xr

from ufs2arco.sources import Source

logger = logging.getLogger("ufs2arco")


class CloudZarrData(Source):
    """
    Generic access patterns for forecast datasets store somewhere in Zarr
    """

    @property
    def available_variables(self) -> tuple:
        return tuple(self._xds.data_vars)

    @property
    def available_levels(self) -> tuple:
        return tuple(self._xds["level"].values)

    def __init__(
        self,
        uri: str,
        variables: Optional[list | tuple] = None,
        levels: Optional[list | tuple] = None,
        use_nearest_levels: Optional[bool] = False,
        slices: Optional[dict] = None,
    ) -> None:

        # open and rename
        xds = xr.open_zarr(
            uri,
            storage_options={"token": "anon"},
            decode_timedelta=True,
        )
        self._xds = xds.rename(self.rename)

        # parent class checks if variables and levels are legit
        super().__init__(
            variables=variables,
            levels=levels,
            use_nearest_levels=use_nearest_levels,
            slices=slices,
        )

        # now subsample the dataset
        self._xds = self._xds[self.variables]
        if self.levels is not None:
            self._xds = self._xds.sel(level=self.levels, **self._level_sel_kwargs)
        self._xds = self.apply_slices(self._xds)


    def open_sample_dataset(
        self,
        dims: dict,
        open_static_vars: bool,
        cache_dir: Optional[str] = None,
    ) -> xr.Dataset:

        xds = self._xds.sel({k: [v] for k, v in dims.items()})
        osv = open_static_vars or self._open_static_vars(dims)
        selection = list(self.variables) if osv else list(self.dynamic_vars)
        xds = xds[selection].copy().load()
        return xds
