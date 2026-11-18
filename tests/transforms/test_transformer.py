import pytest
import xarray as xr

from ufs2arco.transforms.transformer import Transformer


def test_transformer_rejects_unknown_transform():
    with pytest.raises(
        NotImplementedError,
        match=r"transformations are not recognized or not implemented",
    ):
        Transformer(options={"not_real": {}})


def test_transformer_rejects_unknown_mapping():
    with pytest.raises(
        NotImplementedError,
        match=r"mappings are not recognized or not implemented",
    ):
        Transformer(options={"mappings": {"not_real": ["t2m"]}})


def test_transformer_applies_multiply_divide_and_mapping():
    xds = xr.Dataset({"t2m": ("x", [10.0, 12.0])}, coords={"x": [0, 1]})
    tr = Transformer(
        options={
            "multiply": {"t2m": 2},
            "divide": {"t2m": 4},
            "mappings": {"round": ["t2m"]},
        }
    )
    result = tr(xds)
    assert "t2m" not in result
    assert "round_t2m" in result
    assert result["round_t2m"].values.tolist() == [5.0, 6.0]


def test_transformer_dispatches_destagger():
    import numpy as np

    tr = Transformer(options={"destagger": {"U": "west_east"}})
    xds = xr.Dataset(
        {"U": (("south_north", "west_east_stag"), [[0.0, 2.0, 4.0], [10.0, 20.0, 30.0]])},
    )
    out = tr(xds)
    assert out["U"].dims == ("south_north", "west_east")
    np.testing.assert_array_equal(out["U"], [[1.0, 3.0], [15.0, 25.0]])


def test_transformer_dispatches_rotate_vectors():
    import numpy as np

    attrs = {
        "GRIB_gridType": "lambert",
        "GRIB_uvRelativeToGrid": 1,
        "GRIB_latitudeOfFirstGridPointInDegrees": 25.0,
        "GRIB_longitudeOfFirstGridPointInDegrees": 260.0,
        "GRIB_Latin1InDegrees": 25.0,
        "GRIB_Latin2InDegrees": 25.0,
        "GRIB_LoVInDegrees": 262.5,
    }
    u = xr.DataArray(np.ones((2, 2)), dims=("y", "x"), coords={"longitude": ("x", [0.0, 1.0])}, name="u", attrs=attrs)
    v = xr.DataArray(np.ones((2, 2)), dims=("y", "x"), coords={"longitude": ("x", [0.0, 1.0])}, name="v", attrs=attrs)
    xds = xr.Dataset({"u": u, "v": v})
    tr = Transformer(options={"rotate_vectors": {"vector_pairs": [["u", "v"]]}})
    out = tr(xds)
    assert out["u"].attrs["GRIB_uvRelativeToGrid"] == 0


def test_transformer_dispatches_fv_vertical_regrid():
    import numpy as np

    xds = xr.Dataset(
        {
            "delz": (("level",), [1.0, 1.0]),
            "temp": (("level",), [10.0, 20.0]),
        },
        coords={"level": [100, 200]},
    )
    tr = Transformer(
        options={
            "fv_vertical_regrid": {
                "weight_var": "delz",
                "interfaces": [50, 150, 250],
            }
        }
    )
    out = tr(xds)
    assert "temp" in out
    assert "delz" not in out
    assert "vertical_coordinate" in out["temp"].attrs
