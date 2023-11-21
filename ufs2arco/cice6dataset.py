
from typing import Dict, List, Callable
from datetime import datetime
import numpy as np
import xarray as xr

from .ufsdataset import UFSDataset

class CICE6Dataset(UFSDataset):
    __doc__ = UFSDataset.__doc__
    def __init__(self, path_in: Callable, config_filename: str, is_nested: bool = False) -> None:
        super(CICE6Dataset, self).__init__(path_in, config_filename, is_nested)
        self.zarr_name = "cice6.zarr"
        self.chunks_in = self.chunks_in if len(self.chunks_in) != 0 else {
            "nc": 1,
            "nkice": 1,
            "nkaer": 1,
            "nksnow": 1,
            "nj": -1,
            "ni": -1,
        }

        self.chunks_out = self.chunks_out if len(self.chunks_out) != 0 else {
            "time": 1,
            "NCAT": 1,
            "VGRDi": 1,
            "VGRDa": 1,
            "VGRDs": 1,
            "nj": -1,
            "ni": -1,
        }

    def open_dataset(self, cycles: datetime, fsspec_kwargs=None, **kwargs):
        xds = super().open_dataset(cycles, fsspec_kwargs, **kwargs)

        # Deal with time
        # make cftime to be consistent with MOM and FV3
        xds["cftime"] = self._time2cftime(xds["time"])
        xds["ftime"] = self._time2ftime(xds["time"], cycles)
        xds = xds.set_coords(["time", "cftime", "ftime"])

        ## Swap meaningless logical indices for values, where it makes sense (basically, not ni,nj)
        ## Note we can only do this for time_bounds b/c ds is squeezed
        #swap = {
        #    "nc": "NCAT",
        #    "nkaer": "VGRDa",
        #    "nkice": "VGRDi",
        #    "nksnow": "VGRDs",
        #    "d2": "time_bounds",
        #}
        #for org, new in swap.items():
        #    if new in xds and xds[new].ndim>0:
        ##        for d in ["time", "member"]:
        ##            if d in xds[new].dims:
        ##                xds[new] = xds[new].isel({d:0})
        #        xds = xds.swap_dims({org: new})
        #    if org in xds:
        #        xds = xds.drop(org)

        ## This one gets squeezed out and is for some reason problematic?
        #if xds["VGRDs"].ndim == 0:
        #    xds = xds.drop("VGRDs")

        return xds
