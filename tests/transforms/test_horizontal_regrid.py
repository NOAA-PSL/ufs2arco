import numpy as np
import xarray as xr
import importlib

hmod = importlib.import_module("ufs2arco.transforms.horizontal_regrid")

def test_horizontal_regrid_renames_and_drops_bounds(monkeypatch):
    src = xr.Dataset(
        {"a": (("latitude", "longitude"), np.ones((2, 2)))},
        coords={"latitude": [0.0, 1.0], "longitude": [10.0, 11.0]},
    )
    dst = xr.Dataset(coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0]})

    class _FakeRegridder:
        def __init__(self, ds_in, ds_out, **kwargs):
            _ = (ds_in, ds_out, kwargs)

        def __call__(self, xds, keep_attrs=True):
            _ = (xds, keep_attrs)
            out = xr.Dataset(
                {"a": (("lat", "lon"), np.ones((2, 2)))},
                coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0]},
            )
            out["lat_b"] = xr.DataArray([0.0, 0.5, 1.0], dims=("lat_vertices",))
            out["lon_b"] = xr.DataArray([10.0, 10.5, 11.0], dims=("lon_vertices",))
            out["latitude_longitude"] = xr.DataArray(np.nan)
            return out

    monkeypatch.setattr(hmod.xr, "open_dataset", lambda *args, **kwargs: dst)
    monkeypatch.setattr(hmod.os.path, "isfile", lambda _: True)
    monkeypatch.setattr(hmod, "xesmf", type("X", (), {"Regridder": _FakeRegridder}))

    result = hmod.horizontal_regrid(
        src,
        target_grid_path="/tmp/grid.nc",
        regridder_kwargs={"method": "bilinear", "filename": "/tmp/w.nc"},
    )
    assert "latitude" in result.coords
    assert "longitude" in result.coords
    assert "lat_b" not in result
    assert "lon_b" not in result
    assert "latitude_longitude" not in result


def test_get_bounds_adds_lat_b_and_lon_b():
    xds = xr.Dataset(coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0]})
    out = hmod.get_bounds(xds)
    assert "lat_b" in out.coords
    assert "lon_b" in out.coords
