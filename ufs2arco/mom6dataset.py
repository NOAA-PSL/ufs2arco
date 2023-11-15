from datetime import datetime
import numpy as np
import xarray as xr

from .ufsdataset import UFSDataset

class MOM6Dataset(UFSDataset):
    def __init__(self, *args, **kwargs):
        super(MOM6Dataset, self).__init__(*args, **kwargs)
        self.zarr_name = "mom6.zarr"
        self.chunks_in = self.chunks_in if len(self.chunks_in) != 0 else {
            "z_l": 1,
            "z_i": 1,
            "yh": -1,
            "xh": -1,
            "yq": -1,
            "xq": -1,
        }

        self.chunks_out = self.chunks_out if len(self.chunks_out) != 0 else {
            "time": 1,
            "z_l": 1,
            "z_i": 1,
            "yh": -1,
            "xh": -1,
            "yq": -1,
            "xq": -1,
        }

    def open_dataset(self, cycle: datetime, fsspec_kwargs=None, **kwargs):
        xds = super().open_dataset(cycle, fsspec_kwargs, **kwargs)

        # Deal with time
        xds = xds.rename({"time": "cftime"})
        time = self._cftime2time(xds["cftime"])
        xds["time"] = xr.DataArray(
                time,
                coords=xds["cftime"].coords,
                dims=xds["cftime"].dims,
                attrs={
                    "long_name": "time",
                    "axis": "T",
                },
        )
        xds["ftime"] = xr.DataArray(
                time-np.datetime64(cycle),
                coords=xds["cftime"].coords,
                dims=xds["cftime"].dims,
                attrs={
                    "long_name": "forecast_time",
                    "description": f"time passed since {str(cycle)}",
                    "axis": "T",
                },
        )
        xds = xds.swap_dims({"cftime": "time"})
        xds = xds.set_coords(["time", "cftime", "ftime"])
        return xds
