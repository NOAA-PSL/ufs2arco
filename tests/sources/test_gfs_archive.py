import pandas as pd
import pytest

import ufs2arco.sources.noaa_grib_forecast as nmod
from ufs2arco.sources.gfs_archive import GFSArchive


def _varmeta():
    return {
        "t2m": {
            "forecast_only": False,
            "file_suffixes": [""],
            "filter_by_keys": {"typeOfLevel": "surface", "paramId": 1},
            "long_name": "2m temperature",
        },
        "fc_only": {
            "forecast_only": True,
            "file_suffixes": [""],
            "filter_by_keys": {"typeOfLevel": "surface", "paramId": 2},
            "long_name": "forecast only",
        },
    }


def test_gfs_rejects_forecast_only_vars_when_fhr_has_zero(monkeypatch):
    monkeypatch.setattr(nmod.yaml, "safe_load", lambda _: _varmeta())
    with pytest.raises(
        ValueError,
        match=r"requested variables only exist in forecast timesteps",
    ):
        GFSArchive(
            t0={"start": "2020-01-01T00", "end": "2020-01-01T00", "freq": "6h"},
            fhr={"start": 0, "end": 0, "step": 1},
            variables=["fc_only"],
        )


def test_gfs_build_path_switches_legacy_and_aws_branches(monkeypatch):
    monkeypatch.setattr(nmod.yaml, "safe_load", lambda _: _varmeta())
    gfs = GFSArchive(
        t0={"start": "2020-01-01T00", "end": "2020-01-01T00", "freq": "6h"},
        fhr={"start": 1, "end": 1, "step": 1},
        variables=["t2m"],
    )
    old_path = gfs._build_path(pd.Timestamp("2020-01-01T00"), 1, "")
    new_path = gfs._build_path(pd.Timestamp("2022-01-01T12"), 1, "")
    assert "data.rda.ucar.edu" in old_path
    assert "noaa-gfs-bdp-pds" in new_path


def test_gfs_build_path_boundary_for_atmos_segment(monkeypatch):
    monkeypatch.setattr(nmod.yaml, "safe_load", lambda _: _varmeta())
    gfs = GFSArchive(
        t0={"start": "2021-03-22T06", "end": "2021-03-22T06", "freq": "6h"},
        fhr={"start": 1, "end": 1, "step": 1},
        variables=["t2m"],
    )
    at_boundary = gfs._build_path(pd.Timestamp("2021-03-22T06"), 1, "")
    after_boundary = gfs._build_path(pd.Timestamp("2021-03-22T07"), 1, "")
    assert "/atmos/" not in at_boundary
    assert "/atmos/" in after_boundary
