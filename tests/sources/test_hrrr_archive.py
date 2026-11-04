import pandas as pd

import ufs2arco.sources.noaa_grib_forecast as nmod
from ufs2arco.sources.aws_hrrr_archive import AWSHRRRArchive


def _varmeta():
    return {
        "t2m": {
            "forecast_only": False,
            "file_suffixes": ["prs"],
            "filter_by_keys": {"typeOfLevel": "surface", "paramId": 1},
            "long_name": "2m temperature",
        }
    }


def test_hrrr_init_sets_time_and_forecast_hour_axes(monkeypatch):
    monkeypatch.setattr(nmod.yaml, "safe_load", lambda _: _varmeta())
    source = AWSHRRRArchive(
        t0={"start": "2020-01-01T00", "end": "2020-01-01T01", "freq": "1h"},
        fhr={"start": 0, "end": 2, "step": 1},
        variables=["t2m"],
    )
    assert len(source.t0) == 2
    assert source.fhr.tolist() == [0, 1, 2]


def test_hrrr_build_path_format(monkeypatch):
    monkeypatch.setattr(nmod.yaml, "safe_load", lambda _: _varmeta())
    source = AWSHRRRArchive(
        t0={"start": "2020-01-01T00", "end": "2020-01-01T00", "freq": "1h"},
        fhr={"start": 0, "end": 0, "step": 1},
        variables=["t2m"],
    )
    path = source._build_path(
        t0=pd.Timestamp("2020-01-01T06"),
        fhr=7,
        file_suffix="prs",
    )
    assert path == "filecache::s3://noaa-hrrr-bdp-pds/hrrr.20200101/conus/hrrr.t06z.wrfprsf07.grib2"
