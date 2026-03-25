import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ufs2arco.sources.cloud_zarr as cmod
from ufs2arco.sources.cloud_zarr import CloudZarrData


class _CloudLike(CloudZarrData):
    sample_dims = ("time",)
    horizontal_dims = ("latitude", "longitude")
    static_vars = ("lsm",)
    time = pd.date_range("2020-01-01", periods=2, freq="6h")

    @property
    def rename(self):
        return {}


def _fake_cloud_ds():
    return xr.Dataset(
        {
            "lsm": (("time", "latitude", "longitude"), np.ones((2, 2, 2))),
            "t2m": (("time", "latitude", "longitude"), np.ones((2, 2, 2)) * 2),
        },
        coords={
            "time": pd.date_range("2020-01-01", periods=2, freq="6h"),
            "level": ("level", [100]),
            "latitude": [0.0, 1.0],
            "longitude": [10.0, 11.0],
        },
    )


def test_open_sample_dataset(monkeypatch):
    monkeypatch.setattr(cmod.xr, "open_zarr", lambda *args, **kwargs: _fake_cloud_ds())
    src = _CloudLike(uri="s3://dummy", variables=["lsm", "t2m"])
    out = src.open_sample_dataset({"time": src._xds.time.values[1]}, open_static_vars=False)
    assert "t2m" in out
    assert "lsm" not in out
