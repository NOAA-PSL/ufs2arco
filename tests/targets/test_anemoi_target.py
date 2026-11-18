import numpy as np
import pandas as pd
import xarray as xr
import pytest

from ufs2arco.targets.anemoi import Anemoi, AnemoiInferenceWithForcings, _merge_attrs


class _SourceNoFhr:
    sample_dims = ("time",)
    horizontal_dims = ("latitude", "longitude")
    static_vars = tuple()
    time = pd.date_range("2020-01-01", periods=2, freq="6h")
    member = None
    fhr = None

    def add_full_extra_coords(self, xds):
        return xds


class _SourceWithFhr:
    sample_dims = ("t0", "fhr")
    horizontal_dims = ("latitude", "longitude")
    static_vars = tuple()
    t0 = pd.date_range("2020-01-01", periods=2, freq="6h")
    fhr = np.array([0])
    member = None

    def add_full_extra_coords(self, xds):
        return xds


def _target(chunks=None, **kwargs):
    return Anemoi(
        source=_SourceNoFhr(),
        chunks=chunks or {"time": 1, "ensemble": 1, "variable": 1, "cell": 4},
        store_path="/tmp/out.zarr",
        **kwargs,
    )


def test_anemoi_drops_protected_rename_keys():
    tgt = _target(
        chunks={"foo": 1, "ensemble": 1, "variable": 1, "cell": 4},
        rename={"time": "foo"},
    )
    assert "time" not in tgt.rename


def test_sort_channels_by_levels_numeric_ordering():
    tgt = _target(sort_channels_by_levels=True)
    xds = xr.Dataset(
        {
            "gh_100": (("time", "ensemble", "latitudes", "longitudes"), np.ones((1, 1, 1, 1))),
            "gh_1000": (("time", "ensemble", "latitudes", "longitudes"), np.ones((1, 1, 1, 1))),
            "gh_150": (("time", "ensemble", "latitudes", "longitudes"), np.ones((1, 1, 1, 1))),
            "t2m": (("time", "ensemble", "latitudes", "longitudes"), np.ones((1, 1, 1, 1))),
        },
        coords={"time": [0], "ensemble": [0], "latitudes": [0.0], "longitudes": [10.0]},
    )
    out = tgt._stackit(xds)
    assert out.attrs["variables"] == ["gh_100", "gh_150", "gh_1000", "t2m"]


def test_statistics_start_date_raises_if_not_in_datetime():
    tgt = _target(statistics_period={"start": "1999-01-01T00"})
    with pytest.raises(AssertionError, match=r"could not find statistics_start_date within datetime"):
        _ = tgt.statistics_start_date


def test_inference_load_data_flag_and_save_structure():
    target = AnemoiInferenceWithForcings(
        source=_SourceWithFhr(),
        chunks={"time": 1, "ensemble": 1, "variable": 1, "cell": 4},
        store_path="/tmp/out.zarr",
        multistep_input=1,
    )
    assert target.load_data_flag({"t0": _SourceWithFhr.t0[0]})
    assert not target.load_data_flag({"t0": _SourceWithFhr.t0[1]})

    ds = xr.Dataset(
        {
            "t2m": (("t0", "fhr", "latitude", "longitude"), np.ones((1, 1, 2, 2))),
            "lsm": (("t0", "fhr", "latitude", "longitude"), np.ones((1, 1, 2, 2))),
        },
        coords={"t0": [_SourceWithFhr.t0[0]], "fhr": [0], "latitude": [0.0, 1.0], "longitude": [10.0, 11.0]},
    )
    target.save_ds_structure(ds)
    assert "t2m" in target.ds_structure
    assert np.isnan(target.ds_structure["t2m"]).all()
    assert np.isfinite(target.ds_structure["lsm"]).all()


def test_merge_attrs_conflict_raises():
    with pytest.raises(ValueError, match=r"Conflict for common key 'foo'"):
        _merge_attrs(
            [
                {"foo": "a", "variables": ["x"], "variables_metadata": {"x": {}}},
                {"foo": "b", "variables": ["y"], "variables_metadata": {"y": {}}},
            ]
        )
