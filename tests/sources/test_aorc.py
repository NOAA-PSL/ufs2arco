import xarray as xr

from ufs2arco.sources.aws_aorc import AWSAORC


def _fake_aorc_ds(time):
    return xr.Dataset(
        {
            "t2m": (("time", "latitude", "longitude"), [[[1.0, 2.0], [3.0, 4.0]]]),
            "tp": (("time", "latitude", "longitude"), [[[0.1, 0.2], [0.3, 0.4]]]),
        },
        coords={
            "time": [time],
            "latitude": [0.0, 1.0],
            "longitude": [10.0, 11.0],
        },
    )


def test_awsaorc_build_uri(monkeypatch):
    monkeypatch.setattr(
        AWSAORC,
        "_open_and_rename",
        lambda self, time: _fake_aorc_ds(time),
    )
    source = AWSAORC(
        time={"start": "2020-01-01T00", "end": "2020-01-01T00", "freq": "1h"},
        variables=["t2m"],
    )
    uri = source._build_uri(source.time[0])
    assert uri == "s3://noaa-nws-aorc-v1-1-1km/2020.zarr"


def test_awsaorc_open_sample_dataset(monkeypatch):
    monkeypatch.setattr(
        AWSAORC,
        "_open_and_rename",
        lambda self, time: _fake_aorc_ds(time),
    )
    source = AWSAORC(
        time={"start": "2020-01-01T00", "end": "2020-01-01T01", "freq": "1h"},
        variables=["t2m", "tp"],
    )
    out = source.open_sample_dataset(
        dims={"time": source.time[1]},
        open_static_vars=False,
    )
    assert "t2m" in out
    assert "tp" in out
    assert len(out["time"]) == 1
    assert out["time"].values[0] == source.time[1]
