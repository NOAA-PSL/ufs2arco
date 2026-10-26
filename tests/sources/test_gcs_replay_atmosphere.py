import numpy as np
import pandas as pd
import xarray as xr

import ufs2arco.sources.cloud_zarr as cmod
from ufs2arco.sources.gcs_replay_atmosphere import GCSReplayAtmosphere


def _fake_replay_ds():
    return xr.Dataset(
        {
            "t": (("time", "pfull", "grid_yt", "grid_xt"), np.ones((2, 2, 2, 2))),
            "land_static": (("time", "grid_yt", "grid_xt"), np.ones((2, 2, 2))),
            "cftime": (("time",), np.array([0, 1])),
            "ftime": (("time",), np.array([0, 1])),
        },
        coords={
            "time": pd.date_range("2020-01-01", periods=2, freq="6h"),
            "pfull": [500, 1000],
            "grid_yt": [0.0, 1.0],
            "grid_xt": [10.0, 11.0],
        },
    )


def test_replay_renames_and_drops_time_helper_vars(monkeypatch):
    monkeypatch.setattr(cmod.xr, "open_zarr", lambda *args, **kwargs: _fake_replay_ds())
    source = GCSReplayAtmosphere(
        time={"start": "2020-01-01T00", "end": "2020-01-01T06", "freq": "6h"},
        uri="gcs://dummy.zarr",
        variables=["t", "land_static", "cftime", "ftime"],
        levels=[500],
    )
    assert "cftime" not in source._xds
    assert "ftime" not in source._xds
    assert "level" in source._xds.coords
    assert "latitude" in source._xds.coords
    assert "longitude" in source._xds.coords


def test_replay_open_sample_dataset_dynamic_only(monkeypatch):
    monkeypatch.setattr(cmod.xr, "open_zarr", lambda *args, **kwargs: _fake_replay_ds())
    monkeypatch.setattr(
        GCSReplayAtmosphere,
        "static_vars",
        ("land_static", "hgtsfc_static", "cftime", "ftime"),
    )
    source = GCSReplayAtmosphere(
        time={"start": "2020-01-01T00", "end": "2020-01-01T06", "freq": "6h"},
        uri="gcs://dummy.zarr",
        variables=["t", "land_static", "cftime", "ftime"],
        levels=[500],
    )
    out = source.open_sample_dataset(
        dims={"time": source.time[1]},
        open_static_vars=False,
    )
    assert "t" in out
    assert "land_static" not in out
    assert out["level"].values.tolist() == [500]
