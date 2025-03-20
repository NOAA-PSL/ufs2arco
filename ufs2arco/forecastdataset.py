import logging

import pandas as pd
import xarray as xr

from ufs2arco.targetdataset import TargetDataset

logger = logging.getLogger("ufs2arco")

class ForecastDataset(TargetDataset):
    """
    Sort of a default target format.
    Store the data in zarr, essentially how it's laid out originally

    Expected output has dimensions
        ("t0", "fhr", "member", "level", "latitude", "longitude")

    Use the rename argument to modify any of these (and make sure :attr:`chunks` uses those desired names too)
    """

    # THIS IS BAD, because rename is an argument, but these are hard coded attributes
    sample_dims = ("t0", "fhr", "member")
    base_dims = ("level", "latitude", "longitude")

    # Each sample dim needs to be provided here
    # this is a simple case, target is passive to source
    @property
    def t0(self):
        return self.source.t0

    @property
    def fhr(self):
        return self.source.fhr

    @property
    def member(self):
        return self.source.member

    def manage_coords(self, xds: xr.Dataset) -> xr.Dataset:

        # compute the full version of these extra coords
        xds["lead_time"] = xr.DataArray(
            [pd.Timedelta(hours=x) for x in self.fhr],
            coords=xds["fhr"].coords,
            attrs=xds["lead_time"].attrs.copy(),
        )

        xds["valid_time"] = xds["t0"] + xds["lead_time"]
        xds["valid_time"].attrs = xds["valid_time"].attrs.copy()
        xds = xds.set_coords(["lead_time", "valid_time"])
        return xds
