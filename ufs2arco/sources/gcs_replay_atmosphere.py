import logging
from typing import Optional
import yaml

import pandas as pd
import xarray as xr

from ufs2arco.sources import CloudZarrData, Source

logger = logging.getLogger("ufs2arco")


class GCSReplayAtmosphere(CloudZarrData, Source):
    """
    Atmospheric component of replay, already zarr-ified on GCS.
    """

    sample_dims = ("time",)
    horizontal_dims = ("latitude", "longitude")
    static_vars = ("land_static", "hgtsfc_static")

    @property
    def rename(self) -> dict:
        return {
            "pfull": "level",
            "grid_yt": "latitude",
            "grid_xt": "longitude",
        }

    def __init__(
        self,
        time: dict,
        uri: str,
        variables: Optional[list | tuple] = None,
        levels: Optional[list | tuple] = None,
        use_nearest_levels: Optional[bool] = False,
        slices: Optional[dict] = None,
    ) -> None:

        self.time = pd.date_range(**time)

        super().__init__(
            uri=uri,
            variables=variables,
            levels=levels,
            use_nearest_levels=use_nearest_levels,
            slices=slices,
        )
        # drop these because cftime gives trouble no matter what
        self._xds = self._xds.drop_vars(["cftime", "ftime"])
