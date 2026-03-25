import numpy as np
import pytest
import xarray as xr

from ufs2arco.transforms.vertical_regrid import fv_vertical_regrid


def test_vertical_regrid_raises_if_weight_var_missing():
    xds = xr.Dataset({"temp": (("level",), [1.0, 2.0])}, coords={"level": [100, 200]})
    with pytest.raises(AssertionError, match=r"can't find delz in dataset"):
        fv_vertical_regrid(xds, weight_var="delz", interfaces=[50, 250])


def test_vertical_regrid_keep_weight_var_branch():
    xds = xr.Dataset(
        {
            "delz": (("level",), [1.0, 1.0]),
            "temp": (("level",), [10.0, 20.0]),
        },
        coords={"level": [100, 200]},
    )
    out = fv_vertical_regrid(
        xds=xds,
        weight_var="delz",
        interfaces=np.array([50, 150, 250]),
        keep_weight_var=True,
    )
    assert "delz" in out
    assert "temp" in out
    assert "level" in out.dims
    assert out["delz"].attrs.get("vertical_coordinate")


def test_vertical_regrid_raises_if_interfaces_not_found_without_nearest():
    xds = xr.Dataset(
        {
            "delz": (("level",), [1.0, 1.0]),
            "temp": (("level",), [10.0, 20.0]),
        },
        coords={"level": [100, 200], "interface": [50, 150, 250]},
    )
    with pytest.raises(KeyError, match=r"not all values found in index ['\"]interface['\"]"):
        fv_vertical_regrid(
            xds=xds,
            weight_var="delz",
            interfaces=[999, 1000],
            use_nearest_interfaces=False,
        )
