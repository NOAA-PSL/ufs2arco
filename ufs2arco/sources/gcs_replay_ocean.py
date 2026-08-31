import logging
from typing import Optional

import pandas as pd

from ufs2arco.sources import CloudZarrData, Source

logger = logging.getLogger("ufs2arco")


class GCSReplayOcean(CloudZarrData, Source):
    """
    Ocean component of replay, already zarr-ified on GCS.
    """

    sample_dims = ("time",)
    horizontal_dims = ("latitude", "longitude")
    # The MOM6 store has no land_static / hgtsfc_static; those names were copied
    # from the atmosphere source. The ocean equivalent would be "landsea_mask",
    # but it is left as an ordinary variable so that a recipe has to ask for it.
    static_vars = tuple()

    @property
    def rename(self) -> dict:
        return {
            "z_l": "level",
            "lat": "latitude",
            "lon": "longitude",
        }

    def __init__(
        self,
        time: dict,
        uri: str,
        variables: Optional[list | tuple] = None,
        levels: Optional[list | tuple] = None,
        use_nearest_levels: Optional[bool] = True,
        slices: Optional[dict] = None,
        local: Optional[bool] = False,
    ) -> None:
        self.time = pd.date_range(**time)

        super().__init__(
            uri=uri,
            variables=variables,
            levels=levels,
            use_nearest_levels=use_nearest_levels,
            slices=slices,
            local=local,
        )

        # Drop these because cftime gives trouble no matter what.
        self._xds = self._xds.drop_vars(["cftime", "ftime"])
