import logging
from typing import Optional

import pandas as pd
import xarray as xr

from ufs2arco.sources import CloudZarrData, Source

logger = logging.getLogger("ufs2arco")


class GCSERA5OneDegree(CloudZarrData, Source):

    sample_dims = ("time",)
    horizontal_dims = ("latitude", "longitude")
    static_vars = ("land_sea_mask", "geopotential_at_surface")

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
