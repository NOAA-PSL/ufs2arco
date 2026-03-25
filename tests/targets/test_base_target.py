import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ufs2arco.targets.base import Target


class _DummySource:
    sample_dims = ("t0", "fhr")
    horizontal_dims = ("latitude", "longitude")
    t0 = pd.date_range("2020-01-01", periods=1, freq="6h")
    fhr = np.array([0])
    member = None

    def add_full_extra_coords(self, xds):
        xds.attrs["extra_coords_added"] = True
        return xds


def _target():
    return Target(
        source=_DummySource(),
        chunks={"t0": 1, "fhr": 1, "latitude": 2, "longitude": 2},
        store_path="/tmp/out.zarr",
        forcings=["latitude"],
    )


class _DummySourceNoFhr:
    sample_dims = ("time",)
    horizontal_dims = ("latitude", "longitude")
    time = pd.date_range("2020-01-01", periods=2, freq="6h")
    fhr = None
    member = None

    def add_full_extra_coords(self, xds):
        xds.attrs["extra_coords_added"] = True
        return xds


def test_compute_forcings_rejects_dummy_name_collision():
    tgt = _target()
    xds = xr.Dataset(
        {
            "computed_forcing_latitude": (("latitude", "longitude"), np.ones((2, 2))),
        },
        coords={"latitude": [0.0, 1.0], "longitude": [10.0, 11.0], "valid_time": ("t0", _DummySource.t0.values)},
    )
    with pytest.raises(AssertionError, match=r"compute_forcings: computed_forcing_latitude in dataset"):
        tgt.compute_forcings(xds)


def test_rename_dataset_rejects_forcing_name_conflict():
    tgt = _target()
    xds = xr.Dataset(coords={"latitude": [0.0, 1.0], "longitude": [10.0, 11.0]})
    with pytest.raises(AssertionError, match=r"rename_dataset: latitude already in dataset"):
        tgt.rename_dataset(xds)


def test_target_init_rejects_unknown_forcing_name():
    with pytest.raises(NotImplementedError, match=r"requested forcing variable\(s\) \['not_real_forcing'\] are not implemented"):
        Target(
            source=_DummySourceNoFhr(),
            chunks={"time": 1, "latitude": 2, "longitude": 2},
            store_path="/tmp/out.zarr",
            forcings=["not_real_forcing"],
        )


def test_target_init_rejects_non_unit_sample_chunks():
    with pytest.raises(AssertionError, match=r"chunks\['time'\] = 2, but should be 1"):
        Target(
            source=_DummySourceNoFhr(),
            chunks={"time": 2, "latitude": 2, "longitude": 2},
            store_path="/tmp/out.zarr",
        )


def test_apply_transforms_to_sample_computes_and_renames_forcing():
    tgt = Target(
        source=_DummySourceNoFhr(),
        chunks={"time": 1, "latitude": 2, "longitude": 2},
        store_path="/tmp/out.zarr",
        forcings=["cos_latitude"],
    )
    xds = xr.Dataset(
        coords={
            "time": _DummySourceNoFhr.time,
            "latitude": [0.0, 1.0],
            "longitude": [10.0, 11.0],
        }
    )
    out = tgt.apply_transforms_to_sample(xds)
    assert "cos_latitude" in out
    assert "computed_forcing_cos_latitude" not in out


def test_manage_coords_calls_source_add_full_extra_coords():
    tgt = Target(
        source=_DummySourceNoFhr(),
        chunks={"time": 1, "latitude": 2, "longitude": 2},
        store_path="/tmp/out.zarr",
    )
    xds = xr.Dataset(coords={"time": _DummySourceNoFhr.time, "latitude": [0.0, 1.0], "longitude": [10.0, 11.0]})
    out = tgt.manage_coords(xds)
    assert out.attrs["extra_coords_added"] is True
