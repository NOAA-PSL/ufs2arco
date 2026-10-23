import numpy as np
import xarray as xr

from ufs2arco.utils import batched, expand_anemoi_dataset, convert_anemoi_inference_dataset


def test_batched_yields_expected_groups():
    out = list(batched("ABCDEFG", 3))
    assert out == [("A", "B", "C"), ("D", "E", "F"), ("G",)]


def test_batched_raises_for_invalid_n():
    try:
        list(batched([1, 2, 3], 0))
        raise AssertionError("Expected ValueError for n < 1")
    except ValueError:
        pass


def test_expand_anemoi_dataset_groups_levels_and_renames_coords():
    ads = xr.Dataset(
        {
            "data": (
                ("time", "variable", "ensemble", "cell"),
                np.arange(2 * 3 * 1 * 4).reshape(2, 3, 1, 4),
            )
        },
        coords={
            "time": [0, 1],
            "variable": [0, 1, 2],
            "ensemble": [0],
            "cell": [0, 1, 2, 3],
            "dates": ("time", np.array(["2020-01-01", "2020-01-01T06"], dtype="datetime64[s]")),
            "latitudes": ("cell", [0.0, 1.0, 2.0, 3.0]),
            "longitudes": ("cell", [10.0, 11.0, 12.0, 13.0]),
        },
    )
    out = expand_anemoi_dataset(ads, "data", ["gh_100", "gh_500", "t2m"])
    assert "gh" in out
    assert "t2m" in out
    assert "level" in out["gh"].dims
    assert "time" in out.coords
    assert "latitude" in out.coords
    assert "longitude" in out.coords


def test_expand_anemoi_dataset_raises_on_variable_length_mismatch():
    ads = xr.Dataset({"data": (("time", "variable"), np.zeros((1, 2)))}, coords={"time": [0], "variable": [0, 1]})
    try:
        expand_anemoi_dataset(ads, "data", ["only_one"])
        raise AssertionError("Expected ValueError for variable length mismatch")
    except ValueError:
        pass


def test_convert_anemoi_inference_dataset_stacks_levels_and_renames_values():
    xds = xr.Dataset(
        {
            "gh_100": (("time", "ensemble", "values"), np.ones((1, 1, 2))),
            "gh_500": (("time", "ensemble", "values"), np.ones((1, 1, 2)) * 2),
            "t2m": (("time", "ensemble", "values"), np.ones((1, 1, 2)) * 3),
        },
        coords={
            "time": [0],
            "ensemble": [0],
            "values": [0, 1],
            "latitude": ("values", [0.0, 1.0]),
            "longitude": ("values", [10.0, 11.0]),
        },
    )
    out = convert_anemoi_inference_dataset(xds)
    assert "gh" in out
    assert "t2m" in out
    assert "level" in out["gh"].dims
    assert "cell" in out["gh"].dims
