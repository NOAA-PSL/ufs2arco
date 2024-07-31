import os
import yaml
from typing import Dict, List, Callable
from datetime import datetime
import numpy as np
import xarray as xr

from .ufsdataset import UFSDataset

class FV3Dataset(UFSDataset):
    __doc__ = UFSDataset.__doc__
    def __init__(self, path_in: Callable, config_filename: str, is_nested: bool = False) -> None:
        super(FV3Dataset, self).__init__(path_in, config_filename, is_nested)
        self.zarr_name = "fv3.zarr"
        self.chunks_in = self.chunks_in if len(self.chunks_in) != 0 else {
            "pfull": 1,
            "grid_yt": -1,
            "grid_xt": -1,
        }

        self.chunks_out = self.chunks_out if len(self.chunks_out) != 0 else {
            "time": 1,
            "pfull": 1,
            "grid_yt": -1,
            "grid_xt": -1,
        }

    def open_dataset(self, cycles: datetime, fsspec_kwargs=None, **kwargs):
        xds = super().open_dataset(cycles, fsspec_kwargs, **kwargs)

        # Deal with time
        xds = xds.rename({"time": "cftime"})
        xds["time"] = self._cftime2time(xds["cftime"])
        xds["ftime"] = self._time2ftime(xds["time"], cycles)
        xds = xds.swap_dims({"cftime": "time"})
        xds = xds.set_coords(["time", "cftime", "ftime"])

        # convert ak/bk attrs to coordinate arrays
        for key in ["ak", "bk"]:
            if key in xds.attrs:
                xds[key] = xr.DataArray(
                    xds.attrs.pop(key),
                    coords=xds["phalf"].coords,
                    dims=xds["phalf"].dims,
                )
                xds = xds.set_coords(key)

        # rename grid_yt.long_name to avoid typo
        xds["grid_yt"].attrs["long_name"] = "T-cell latitude"
        return xds


