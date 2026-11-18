import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ufs2arco.sources import CloudZarrData, Source
from ufs2arco.sources.gcs_replay_atmosphere import GCSReplayAtmosphere
from ufs2arco.sources.gcs_replay_ocean import GCSReplayOcean


class TinyCloudZarr(CloudZarrData, Source):
    sample_dims = ("time",)
    horizontal_dims = ("latitude", "longitude")
    static_vars = ("static",)

    def __init__(self, time, uri, **kwargs):
        self.time = pd.date_range(**time)
        super().__init__(uri=uri, **kwargs)


def test_replay_ocean_does_not_request_atmospheric_static_variables():
    assert GCSReplayOcean.static_vars == tuple()


def test_replay_atmosphere_accepts_store_without_auxiliary_time_coords(
    tmp_path,
):
    store = tmp_path / "analysis.zarr"
    xds = xr.Dataset(
        {
            "tmp": (
                ("time", "pfull", "grid_yt", "grid_xt"),
                np.ones((1, 1, 1, 1), dtype=np.float32),
            ),
        },
        coords={
            "time": pd.to_datetime(["2020-01-01T00"]),
            "pfull": [25.0],
            "grid_yt": [0.0],
            "grid_xt": [0.0],
        },
    )
    xds.to_zarr(store)

    source = GCSReplayAtmosphere(
        time={"start": "2020-01-01T00", "periods": 1, "freq": "6h"},
        uri=str(store),
        variables=["tmp"],
        local=True,
    )

    sample = source.open_sample_dataset(
        dims={"time": pd.Timestamp("2020-01-01T00")},
        open_static_vars=False,
    )
    assert sample.tmp.item() == 1.0


def write_tiny_zarr(tmp_path):
    store = tmp_path / "tiny.zarr"
    time = pd.date_range("2022-01-01T00", periods=5, freq="3h")
    write_zarr_at_times(store, time)
    return store


def write_zarr_at_times(store, time, values=None):
    latitude = [40.0]
    longitude = [250.0]
    if values is None:
        values = np.arange(1, len(time) + 1, dtype=np.float32).reshape(len(time), 1, 1)
    else:
        values = np.array(values, dtype=np.float32).reshape(len(time), 1, 1)
    xds = xr.Dataset(
        data_vars={
            "tprcp": (
                ("time", "latitude", "longitude"),
                values,
                {"units": "kg m-2"},
            ),
            "tmp": (
                ("time", "latitude", "longitude"),
                values * 10,
                {"units": "K"},
            ),
            "static": (
                ("latitude", "longitude"),
                np.array([[7.0]], dtype=np.float32),
            ),
        },
        coords={
            "time": time,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    xds.to_zarr(store)


def test_cloud_zarr_preserves_exact_time_selection_without_accumulation(tmp_path):
    store = write_tiny_zarr(tmp_path)
    source = TinyCloudZarr(
        time={"start": "2022-01-01T06", "periods": 2, "freq": "6h"},
        uri=str(store),
        variables=["tprcp", "tmp"],
        local=True,
    )

    xds = source.open_sample_dataset(
        dims={"time": pd.Timestamp("2022-01-01T06")},
        open_static_vars=False,
    )

    assert xds.sizes["time"] == 1
    assert pd.Timestamp(xds.time.values[0]) == pd.Timestamp("2022-01-01T06")
    assert xds.tprcp.item() == 3.0
    assert xds.tmp.item() == 30.0


def test_cloud_zarr_accumulates_selected_variables_over_requested_window(tmp_path):
    store = write_tiny_zarr(tmp_path)
    source = TinyCloudZarr(
        time={"start": "2022-01-01T06", "periods": 2, "freq": "6h"},
        uri=str(store),
        variables=["tprcp", "tmp", "static"],
        local=True,
        accumulate_variables=["tprcp"],
    )

    first = source.open_sample_dataset(
        dims={"time": pd.Timestamp("2022-01-01T06")},
        open_static_vars=True,
    )
    second = source.open_sample_dataset(
        dims={"time": pd.Timestamp("2022-01-01T12")},
        open_static_vars=False,
    )

    assert first.sizes["time"] == 1
    assert pd.Timestamp(first.time.values[0]) == pd.Timestamp("2022-01-01T06")
    assert first.tprcp.item() == 5.0
    assert first.tmp.item() == 30.0
    assert first.static.item() == 7.0

    assert second.sizes["time"] == 1
    assert pd.Timestamp(second.time.values[0]) == pd.Timestamp("2022-01-01T12")
    assert second.tprcp.item() == 9.0
    assert second.tmp.item() == 50.0
    assert "static" not in second

    assert second.tprcp.attrs["cell_methods"] == "time: sum"
    assert second.tprcp.attrs["accumulation_duration"] == "P0DT6H0M0S"
    assert second.tprcp.attrs["accumulation_window"] == (
        "(2022-01-01T06:00:00, 2022-01-01T12:00:00]"
    )


def test_cloud_zarr_rejects_partial_accumulation_windows(tmp_path):
    store = tmp_path / "gappy.zarr"
    write_zarr_at_times(
        store,
        pd.to_datetime(
            [
                "2022-01-01T00",
                "2022-01-01T03",
                "2022-01-01T09",
                "2022-01-01T12",
            ]
        ),
    )
    source = TinyCloudZarr(
        time={"start": "2022-01-01T09", "periods": 2, "freq": "6h"},
        uri=str(store),
        variables=["tprcp"],
        local=True,
        accumulate_variables=["tprcp"],
    )

    with pytest.raises(ValueError, match="expected 2"):
        source.open_sample_dataset(
            dims={"time": pd.Timestamp("2022-01-01T09")},
            open_static_vars=False,
        )


def test_cloud_zarr_rejects_wrong_timestamp_accumulation_windows(tmp_path):
    store = tmp_path / "irregular.zarr"
    write_zarr_at_times(
        store,
        pd.to_datetime(
            [
                "2022-01-01T00",
                "2022-01-01T03",
                "2022-01-01T07",
                "2022-01-01T12",
            ]
        ),
    )
    source = TinyCloudZarr(
        time={"start": "2022-01-01T12", "periods": 1, "freq": "6h"},
        uri=str(store),
        variables=["tprcp"],
        local=True,
        accumulate_variables=["tprcp"],
    )

    with pytest.raises(ValueError, match="found source times"):
        source.open_sample_dataset(
            dims={"time": pd.Timestamp("2022-01-01T12")},
            open_static_vars=False,
        )


def test_cloud_zarr_accumulation_preserves_nan_buckets(tmp_path):
    store = tmp_path / "with-nan.zarr"
    write_zarr_at_times(
        store,
        pd.date_range("2022-01-01T00", periods=3, freq="3h"),
        values=[1.0, np.nan, 3.0],
    )
    source = TinyCloudZarr(
        time={"start": "2022-01-01T06", "periods": 1, "freq": "6h"},
        uri=str(store),
        variables=["tprcp"],
        local=True,
        accumulate_variables=["tprcp"],
    )

    xds = source.open_sample_dataset(
        dims={"time": pd.Timestamp("2022-01-01T06")},
        open_static_vars=False,
    )

    assert np.isnan(xds.tprcp.item())


def test_cloud_zarr_static_only_non_static_sample_returns_empty_dataset(tmp_path):
    store = write_tiny_zarr(tmp_path)
    source = TinyCloudZarr(
        time={"start": "2022-01-01T06", "periods": 2, "freq": "6h"},
        uri=str(store),
        variables=["static"],
        local=True,
    )

    xds = source.open_sample_dataset(
        dims={"time": pd.Timestamp("2022-01-01T12")},
        open_static_vars=False,
    )

    assert len(xds.data_vars) == 0


def test_cloud_zarr_returns_empty_dataset_for_missing_exact_time(tmp_path):
    store = tmp_path / "gappy-exact.zarr"
    write_zarr_at_times(
        store,
        pd.to_datetime(
            [
                "2022-01-01T00",
                "2022-01-01T06",
                "2022-01-01T18",
            ]
        ),
    )
    source = TinyCloudZarr(
        time={"start": "2022-01-01T00", "periods": 4, "freq": "6h"},
        uri=str(store),
        variables=["tmp"],
        local=True,
    )

    xds = source.open_sample_dataset(
        dims={"time": pd.Timestamp("2022-01-01T12")},
        open_static_vars=False,
    )

    assert len(xds) == 0
    assert "time" not in xds.sizes


def test_cloud_zarr_selects_julian_time_by_unchanged_components(tmp_path):
    cftime = pytest.importorskip("cftime")
    store = tmp_path / "julian.zarr"
    source_times = xr.CFTimeIndex(
        [
            cftime.DatetimeJulian(2022, 1, 1, 0),
            cftime.DatetimeJulian(2022, 1, 1, 6),
        ]
    )
    write_zarr_at_times(store, source_times, values=[1.0, 2.0])
    source = TinyCloudZarr(
        time={"start": "2022-01-01T00", "periods": 2, "freq": "6h"},
        uri=str(store),
        variables=["tmp"],
        local=True,
    )

    xds = source.open_sample_dataset(
        dims={"time": pd.Timestamp("2022-01-01T06")},
        open_static_vars=False,
    )

    assert xds.sizes["time"] == 1
    assert np.issubdtype(xds.time.dtype, np.datetime64)
    assert pd.Timestamp(xds.time.values[0]) == pd.Timestamp("2022-01-01T06")
    assert xds.tmp.item() == 20.0


def test_cloud_zarr_returns_empty_dataset_for_missing_julian_time(tmp_path):
    cftime = pytest.importorskip("cftime")
    store = tmp_path / "gappy-julian.zarr"
    source_times = xr.CFTimeIndex(
        [
            cftime.DatetimeJulian(2022, 1, 1, 0),
            cftime.DatetimeJulian(2022, 1, 1, 12),
        ]
    )
    write_zarr_at_times(store, source_times)
    source = TinyCloudZarr(
        time={"start": "2022-01-01T00", "periods": 3, "freq": "6h"},
        uri=str(store),
        variables=["tmp"],
        local=True,
    )

    xds = source.open_sample_dataset(
        dims={"time": pd.Timestamp("2022-01-01T06")},
        open_static_vars=False,
    )

    assert len(xds) == 0
