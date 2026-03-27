import pytest
import xarray as xr
import pandas as pd

from ufs2arco.sources.base import Source


class _DummySource(Source):
    sample_dims = ("time",)
    horizontal_dims = ("latitude", "longitude")
    static_vars = ("lsm",)
    available_variables = ("t2m", "lsm")
    available_levels = (100, 500)
    time = pd.date_range("2020-01-01", periods=2, freq="6h")


def test_source_rejects_unrecognized_variable():
    with pytest.raises(
        NotImplementedError,
        match=r"variables are not recognized or not implemented",
    ):
        _DummySource(variables=["bad"])


def test_source_accepts_unknown_levels_when_use_nearest():
    src = _DummySource(variables=["t2m"], levels=[925], use_nearest_levels=True)
    assert src._level_sel_kwargs == {"method": "nearest"}


def test_source_rejects_unknown_levels_without_nearest():
    with pytest.raises(ValueError, match=r"use_nearest_levels=True"):
        _DummySource(variables=["t2m"], levels=[925], use_nearest_levels=False)


def test_source_rejects_unknown_slice_key():
    with pytest.raises(NotImplementedError, match=r"only \('sel', 'isel'\) are recognized"):
        _DummySource(variables=["t2m"], slices={"bad_slice": {"latitude": (0, 1)}})


def test_apply_slices_sel_then_isel():
    src = _DummySource(
        variables=["t2m"],
        slices={"sel": {"latitude": (0, 2)}, "isel": {"longitude": (0, 2)}},
    )
    xds = xr.Dataset(
        {"t2m": (("latitude", "longitude"), [[1, 2, 3], [4, 5, 6], [7, 8, 9]])},
        coords={"latitude": [0, 1, 2], "longitude": [10, 11, 12]},
    )
    out = src.apply_slices(xds)
    assert list(out.latitude.values) == [0, 1, 2]
    assert list(out.longitude.values) == [10, 11]



