import logging
from typing import Optional

import pandas as pd
import xarray as xr

import planetary_computer
import pystac_client

from ufs2arco.sources import CloudZarrData, Source

logger = logging.getLogger("ufs2arco")


class CONUS404(CloudZarrData, Source):
    '''
    https://planetarycomputer.microsoft.com/dataset/conus404 

    Temporal Range: 1979-10-01T00:00:00Z, 2022-09-30T23:00:00Z
    Latitude Range: 17.647308349609375, 57.34341812133789
    Longitude Range: -138.73135375976562, -57.068634033203125
    '''

    sample_dims = ("time",)
    horizontal_dims = ("latitude", "longitude")

    asset = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    ).get_collection("conus404").assets["zarr-abfs"]
    storage_options = asset.extra_fields["xarray:storage_options"]

    @property
    def available_levels(self) -> tuple:
        return (
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
            41, 42, 43, 44, 45, 46, 47, 48, 49, 50
        )

    @property
    def rename(self) -> dict:
        return {
            "bottom_top": "level",
            "bottom_top_stag": "level",
            "south_north": "latitude",
            "west_east": "longitude"
        }


    def __init__(
        self,
        time: dict,
        uri: str,
        variables: Optional[list | tuple] = None,
        levels: Optional[list | tuple] = None,
        use_nearest_levels: Optional[bool] = False,
        slices: Optional[dict] = None,
        storage_options: Optional[str] = storage_options,
    ) -> None:

        self.time = pd.date_range(**time)

        super().__init__(
            uri=uri,
            variables=variables,
            levels=levels,
            use_nearest_levels=use_nearest_levels,
            slices=slices,
            storage_options=storage_options,
        )
