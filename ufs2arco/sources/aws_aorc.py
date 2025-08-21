import logging
from typing import Optional
import yaml

import pandas as pd
import xarray as xr

from ufs2arco.sources import Source

logger = logging.getLogger("ufs2arco")


class AWSAORC(Source):
    """
    NOAA Analysis of Record for Calibration (AORC) Dataset available through AWS

    For more information, check out `this page <https://aws.amazon.com/marketplace/pp/prodview-m2sp7gsk5ts6s#resources>`_
    """

    sample_dims = ("time",)
    horizontal_dims = ("latitude", "longitude")
    static_vars = tuple()

    @property
    def available_variables(self) -> tuple:
        return tuple(self._xds.data_vars)

    @property
    def available_levels(self) -> tuple:
        return None

    def __init__(
        self,
        time: dict,
        variables: Optional[list | tuple] = None,
        levels: Optional[list | tuple] = None,
        use_nearest_levels: Optional[bool] = False,
        slices: Optional[dict] = None,
    ) -> None:

        self.time = pd.date_range(**time)

        # open first year to query variables and rename
        self._xds = self._open_and_rename(self.time[0])

        # parent class checks if variables and levels are legit
        super().__init__(
            variables=variables,
            levels=levels,
            use_nearest_levels=use_nearest_levels,
            slices=slices,
        )

        # now subsample the dataset
        self._xds = self._subsample(self._xds)


    def open_sample_dataset(
        self,
        dims: dict,
        open_static_vars: bool,
        cache_dir: Optional[str] = None,
    ) -> xr.Dataset:

        xds = self._open_and_rename(**dims)
        xds = self._subsample(xds)
        xds = xds.sel({k: [v] for k, v in dims.items()})
        osv = open_static_vars or self._open_static_vars(dims)
        selection = list(self.variables) if osv else list(self.dynamic_vars)
        xds = xds[selection].copy().load()
        return xds


    def _open_and_rename(self, time: pd.Timestamp) -> xr.Dataset:

        xds = xr.open_zarr(
            self._build_uri(time),
            storage_options={"s3": {"anon": True}},
            decode_timedelta=True,
        )
        return xds.rename(self.rename)


    def _subsample(self, xds: xr.Dataset) -> xr.Dataset:

        xds = xds[self.variables]
        if self.levels is not None:
            xds = xds.sel(level=self.levels, **self._level_sel_kwargs)
        xds = self.apply_slices(xds)
        return xds


    def _build_uri(self, time: pd.Timestamp) -> str:
        return f"s3://noaa-nws-aorc-v1-1-1km/{time.year}.zarr"

