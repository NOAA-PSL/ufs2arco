import numpy as np
import xarray as xr

from ufs2arco.transforms.mappings import get_available_mappings, apply_mappings


def test_get_available_mappings_contains_expected_entries():
    mappings = get_available_mappings()
    assert set(mappings.keys()) == {"log", "round"}


def test_apply_mappings_supports_string_and_list_inputs():
    xds = xr.Dataset(
        {
            "a": xr.DataArray([1.2, 2.6], dims=("x",), attrs={"long_name": "field a", "units": "m"}),
            "b": xr.DataArray([1.0, 10.0], dims=("x",), attrs={"long_name": "field b", "units": "kg"}),
        },
        coords={"x": [0, 1]},
    )
    out = apply_mappings(xds, {"round": "a", "log": ["b"]})
    assert "a" not in out
    assert "b" not in out
    assert "round_a" in out
    assert "log_b" in out
    assert out["round_a"].values.tolist() == [1.0, 3.0]
    np.testing.assert_allclose(out["log_b"].values, np.log([1.0, 10.0]))
    assert out["round_a"].attrs["long_name"] == "round of field a"
    assert out["log_b"].attrs["long_name"] == "log of field b"


def test_apply_mappings_protected_log_non_positive_to_zero_and_units_empty():
    xds = xr.Dataset(
        {
            "q": xr.DataArray([-2.0, 0.0, 2.0], dims=("x",), attrs={"units": "kg/kg", "long_name": "humidity"}),
        },
        coords={"x": [0, 1, 2]},
    )
    out = apply_mappings(xds, {"log": ["q"]})
    assert "log_q" in out
    assert out["log_q"].values.tolist()[0:2] == [0.0, 0.0]
    assert np.isclose(out["log_q"].values.tolist()[2], np.log(2.0))
    assert out["log_q"].attrs["units"] == ""


def test_apply_mappings_ignores_missing_variable_names():
    xds = xr.Dataset({"a": ("x", [1.0, 2.0])}, coords={"x": [0, 1]})
    out = apply_mappings(xds, {"round": ["not_here"]})
    assert "a" in out
    assert "round_not_here" not in out
