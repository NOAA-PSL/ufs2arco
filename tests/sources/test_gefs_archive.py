import pandas as pd
import numpy as np
import xarray as xr

import ufs2arco.sources.noaa_grib_forecast as nmod
from ufs2arco.sources.aws_gefs_archive import AWSGEFSArchive


def _varmeta():
    return {
        "tmp": {
            "file_suffixes": ["a"],
            "filter_by_keys": {"typeOfLevel": "surface", "paramId": 1},
            "long_name": "tmp",
            "forecast_only": False,
        }
    }


def test_gefs_init_name_and_axes(monkeypatch):
    monkeypatch.setattr(nmod.yaml, "safe_load", lambda _: _varmeta())
    gefs = AWSGEFSArchive(
        t0={"start": "2017-01-01T00", "end": "2017-01-01T18", "freq": "6h"},
        fhr={"start": 0, "end": 6, "step": 6},
        member={"start": 0, "end": 1, "step": 1},
        variables=["tmp"],
    )
    assert gefs.name == "AWSGEFSArchive"
    assert len(gefs.t0) == 4
    assert np.array_equal(gefs.fhr, np.array([0, 6]))
    assert np.array_equal(gefs.member, np.array([0, 1]))
    assert isinstance(str(gefs), str)


def test_gefs_open_sample_dataset_unit_path(monkeypatch):
    monkeypatch.setattr(nmod.yaml, "safe_load", lambda _: _varmeta())
    gefs = AWSGEFSArchive(
        t0={"start": "2017-01-01T00", "end": "2017-01-01T00", "freq": "6h"},
        fhr={"start": 0, "end": 0, "step": 1},
        member={"start": 0, "end": 0, "step": 1},
        variables=["tmp"],
    )

    monkeypatch.setattr(gefs, "_open_local", lambda dims, suffix, cache_dir: "/tmp/mock.grib2")
    monkeypatch.setattr(
        gefs,
        "_open_single_variable",
        lambda dims, varname, file: xr.DataArray(
            np.ones((1, 1, 1, 2, 2)),
            dims=("t0", "fhr", "member", "latitude", "longitude"),
            coords={
                "t0": [dims["t0"]],
                "fhr": [dims["fhr"]],
                "member": [dims["member"]],
                "latitude": [0.0, 1.0],
                "longitude": [10.0, 11.0],
            },
            name=varname,
        ),
    )

    out = gefs.open_sample_dataset(
        dims={"t0": pd.Timestamp("2017-01-01T00"), "fhr": 0, "member": 0},
        open_static_vars=True,
        cache_dir="/tmp/cache",
    )
    assert isinstance(out, xr.Dataset)
    assert "tmp" in out


def test_gefs_build_path(monkeypatch):
    monkeypatch.setattr(nmod.yaml, "safe_load", lambda _: _varmeta())
    gefs = AWSGEFSArchive(
        t0={"start": "2017-01-01T00", "end": "2017-01-01T00", "freq": "6h"},
        fhr={"start": 0, "end": 0, "step": 1},
        member={"start": 0, "end": 0, "step": 1},
        variables=["tmp"],
    )
    p1 = gefs._build_path(pd.Timestamp("2017-01-01T00"), 0, 0, "a")
    p2 = gefs._build_path(pd.Timestamp("2019-01-01T00"), 1, 6, "a")
    p3 = gefs._build_path(pd.Timestamp("2022-01-01T00"), 1, 6, "a")
    assert "pgrb2a/" not in p1 and ".0p50." not in p1
    assert "pgrb2a/" in p2
    assert "atmos/pgrb2ap5/" in p3 and ".0p50." in p3


def test_gefs_build_path_boundaries(monkeypatch):
    monkeypatch.setattr(nmod.yaml, "safe_load", lambda _: _varmeta())
    gefs = AWSGEFSArchive(
        t0={"start": "2018-07-27T00", "end": "2018-07-27T00", "freq": "6h"},
        fhr={"start": 0, "end": 0, "step": 1},
        member={"start": 1, "end": 1, "step": 1},
        variables=["tmp"],
    )
    at_20180727 = gefs._build_path(pd.Timestamp("2018-07-27T00"), 1, 6, "a")
    at_20200923_12 = gefs._build_path(pd.Timestamp("2020-09-23T12"), 1, 6, "a")
    assert "pgrb2a/" in at_20180727
    assert "atmos/pgrb2ap5/" in at_20200923_12
