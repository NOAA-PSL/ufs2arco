import numpy as np
import xarray as xr
import importlib

hmod = importlib.import_module("ufs2arco.transforms.horizontal_regrid")
import pytest
from ufs2arco.regrid.gaussian_grid import gaussian_latitudes

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


def test_get_bounds_rectilinear_vertex_count_and_values():
    # 2 points -> 3 vertices (n+1); evenly spaced cell-centered bounds
    xds = xr.Dataset(coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0]})
    out = hmod.get_bounds(xds)
    assert "lat_b" in out.coords
    assert "lon_b" in out.coords
    assert out["lat_b"].size == 3
    assert out["lon_b"].size == 3
    # cf_xarray cell bounds for [0, 1] -> [-0.5, 0.5, 1.5]
    np.testing.assert_allclose(np.sort(out["lat_b"].values), [-0.5, 0.5, 1.5])
    np.testing.assert_allclose(np.sort(out["lon_b"].values), [9.5, 10.5, 11.5])


def test_get_bounds_gaussian_uses_gaussian_latitudes():
    # build a real (small) gaussian grid: n=2 -> 4 latitudes
    n = 2
    lats, _ = gaussian_latitudes(n)
    xds = xr.Dataset(coords={"lat": lats, "lon": np.array([0.0, 90.0, 180.0, 270.0])})

    out = hmod.get_bounds(xds, is_gaussian=True)

    # lat_b should have len(lat)+1 vertices
    assert out["lat_b"].size == len(lats) + 1
    assert out["lat_b"].dims == ("lat_vertices",)

    # the gaussian branch recomputes lat_b from gaussian_latitudes bounds
    _, lat_b2d = gaussian_latitudes(len(lats) // 2)
    expected = np.concatenate([lat_b2d[:, 0], [lat_b2d[-1, -1]]])
    # lats here are ascending (negative first), so no reversal expected
    np.testing.assert_allclose(out["lat_b"].values, expected, atol=1e-9)


def test_get_bounds_gaussian_reverses_for_descending_latitudes():
    # descending latitudes (positive first) should trigger the reversal branch
    n = 2
    lats, _ = gaussian_latitudes(n)
    lats_desc = lats[::-1]
    xds = xr.Dataset(coords={"lat": lats_desc, "lon": np.array([0.0, 180.0])})

    out = hmod.get_bounds(xds, is_gaussian=True)
    assert out["lat_b"].size == len(lats_desc) + 1
    # first bound should now be positive (reversed) since lat[0] > 0
    assert out["lat"].values[0] > 0
    assert out["lat_b"].values[0] > out["lat_b"].values[-1]


def test_gaussian_latitudes_symmetry_and_count():
    n = 4
    lats, bounds = gaussian_latitudes(n)
    assert lats.size == 2 * n
    assert bounds.shape == (2 * n, 2)
    # latitudes are symmetric about the equator
    np.testing.assert_allclose(lats, -lats[::-1], atol=1e-9)
    # outer bounds span the poles
    assert np.isclose(bounds[0, 0], -90.0)
    assert np.isclose(bounds[-1, -1], 90.0)


def test_gaussian_latitudes_rejects_negative():
    with pytest.raises(ValueError, match="non-negative integer"):
        gaussian_latitudes(-1)
