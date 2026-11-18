import numpy as np
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
