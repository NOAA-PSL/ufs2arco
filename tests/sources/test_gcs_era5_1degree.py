import numpy as np
import pandas as pd
import xarray as xr

import ufs2arco.sources.cloud_zarr as cmod
from ufs2arco.sources.gcs_era5_1degree import GCSERA5OneDegree


def _fake_era5_ds():
    return xr.Dataset(
        {
            "t": (("time", "level", "latitude", "longitude"), np.ones((2, 2, 2, 2))),
            "land_sea_mask": (("time", "latitude", "longitude"), np.ones((2, 2, 2))),
        },
        coords={
            "time": pd.date_range("2020-01-01", periods=2, freq="6h"),
            "level": [500, 1000],
            "latitude": [0.0, 1.0],
            "longitude": [10.0, 11.0],
        },
    )


def test_era5_init_and_dynamic_open_sample_dataset(monkeypatch):
    monkeypatch.setattr(cmod.xr, "open_zarr", lambda *args, **kwargs: _fake_era5_ds())
    source = GCSERA5OneDegree(
        time={"start": "2020-01-01T00", "end": "2020-01-01T06", "freq": "6h"},
        uri="gcs://dummy.zarr",
        variables=["t", "land_sea_mask"],
        levels=[500],
    )
    out = source.open_sample_dataset(
        dims={"time": source.time[1]},
        open_static_vars=False,
    )
    assert "t" in out
    assert "land_sea_mask" not in out
    assert out["level"].values.tolist() == [500]
