import numpy as np
import pytest
import xarray as xr

from ufs2arco.transforms.destagger import destagger


def _dataset():
    return xr.Dataset(
        {
            "U": (("south_north", "west_east_stag"), [[0.0, 2, 4, 6], [10, 20, 30, 40]]),
            "T": (("south_north", "west_east"), np.zeros((2, 3))),
        }
    )


def test_destagger_averages_adjacent_points_and_drops_stag_dim():
    out = destagger(_dataset(), {"U": "west_east"})

    assert out["U"].dims == ("south_north", "west_east")
    np.testing.assert_array_equal(out["U"], [[1.0, 3, 5], [15, 25, 35]])
    assert "west_east_stag" not in out.dims


def test_destagger_skips_variable_without_stag_dim():
    xds = _dataset()
    out = destagger(xds, {"T": "west_east"})

    xr.testing.assert_identical(out["T"], xds["T"])
    assert "west_east_stag" in out.dims


def test_destagger_does_not_modify_input():
    xds = _dataset()
    destagger(xds, {"U": "west_east"})

    xr.testing.assert_identical(xds, _dataset())


def _level_dataset(levels=None):
    z = np.arange(6 * 2, dtype=float).reshape(6, 2) ** 2
    xds = xr.Dataset(
        {"Z": (("level_stag", "x"), z, {"units": "m"})},
        coords={"level_stag": np.arange(6)},
    )
    if levels is not None:
        levels = np.asarray(levels)
        xds = xds.sel(level_stag=np.union1d(levels, levels + 1)).assign_coords(level=levels)
    return xds, z


def test_destagger_level_without_coordinates_uses_full_column():
    z = np.random.default_rng(0).normal(size=(1, 51, 2, 3))
    xds = xr.Dataset({"Z": (("time", "level_stag", "latitude", "longitude"), z)})

    out = destagger(xds, {"Z": "level"})

    assert out["Z"].dims == ("time", "level", "latitude", "longitude")
    np.testing.assert_allclose(out["Z"].values, 0.5 * (z[:, :-1] + z[:, 1:]))


def test_destagger_level_pairs_interfaces_by_label_for_subset():
    levels = [0, 2, 4]
    xds, z = _level_dataset(levels)

    out = destagger(xds, {"Z": "level"})

    assert out["Z"].dims == ("level", "x")
    np.testing.assert_array_equal(out["level"], levels)
    np.testing.assert_allclose(out["Z"].values, 0.5 * (z[:-1] + z[1:])[levels])
    assert out["Z"].attrs == {"units": "m"}
    assert "level_stag" not in out.dims


def test_destagger_level_missing_upper_interface_raises():
    xds, _ = _level_dataset([0, 2])
    xds = xds.sel(level_stag=[0, 1, 2])

    with pytest.raises(KeyError):
        destagger(xds, {"Z": "level"})


def test_destagger_horizontal_rejects_inconsistent_subset():
    xds = _dataset().isel(west_east=slice(0, 2))

    with pytest.raises(ValueError, match="expected west_east_stag = west_east \\+ 1"):
        destagger(xds, {"U": "west_east"})
