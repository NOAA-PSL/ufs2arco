import logging
from typing import Optional
import yaml

import pandas as pd
import xarray as xr

from ufs2arco.sources import AnalysisSource

logger = logging.getLogger("ufs2arco")


class GCSReplayAtmosphere(AnalysisSource):
    """
    Atmospheric component of replay, already zarr-ified on GCS.

    These zarr stores could very easily be generalized... I'm sure.
    """

    static_vars = ("land_static", "hgtsfc_static")

    _is_selected = False

    @property
    def rename(self) -> dict:
        return {
            "pfull": "level",
            "grid_yt": "latitude",
            "grid_xt": "longitude",
        }

    @property
    def available_variables(self) -> tuple:
        return tuple(self._xds.data_vars)

    @property
    def available_levels(self) -> tuple:
        return tuple(self._xds["level"].values)

    def __init__(
        self,
        uri: str,
        time: dict,
        variables: Optional[list | tuple] = None,
        levels: Optional[list | tuple] = None,
        use_nearest_levels: Optional[bool] = False,
    ) -> None:

        # open and rename
        xds = xr.open_zarr(
            uri,
            storage_options={"token": "anon"},
        )
        self._xds = xds.rename(self.rename)

        # parent class checks if variables and levels are legit
        super().__init__(
            time=time,
            variables=variables,
            levels=levels,
            use_nearest_levels=use_nearest_levels,
        )

        # now subsample the dataset
        self._xds = self._xds[self.variables]
        self._xds = self._xds.sel(level=self.levels, **self._level_sel_kwargs)

        # drop these because cftime gives trouble no matter what
        self._xds = self._xds.drop_vars(["cftime", "ftime"])

    def open_sample_dataset(
        self,
        time: pd.Timestamp,
        open_static_vars: bool,
        cache_dir: Optional[str] = None,
    ) -> xr.Dataset:

        xds = self._xds.sel(time=[time])
        selection = list(self.variables) if open_static_vars else list(self.dynamic_vars)
        xds = xds[selection].copy().load()
        return xds
