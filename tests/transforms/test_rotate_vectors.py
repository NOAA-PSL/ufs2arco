import numpy as np
import pytest
import xarray as xr

from ufs2arco.transforms.rotate_vectors import rotate_lambert_conical_vectors, rotate_vectors

LOV = 262.5


def _attrs(latin1=25.0, latin2=25.0):
    return {
        "GRIB_gridType": "lambert",
        "GRIB_uvRelativeToGrid": 1,
        "GRIB_latitudeOfFirstGridPointInDegrees": 25.0,
        "GRIB_longitudeOfFirstGridPointInDegrees": 260.0,
        "GRIB_Latin1InDegrees": latin1,
        "GRIB_Latin2InDegrees": latin2,
        "GRIB_LoVInDegrees": LOV,
    }


def _pair(longitude, attrs=None, u=1.0, v=0.0):
    kw = {
        "dims": "x",
        "coords": {"longitude": ("x", np.asarray(longitude, dtype=float))},
        "attrs": attrs or _attrs(),
    }
    n = len(longitude)
    return (
        xr.DataArray(np.full(n, u), name="u", **kw),
        xr.DataArray(np.full(n, v), name="v", **kw),
    )


def _rotation_deg(u_east, v_north):
    """Clockwise rotation applied to a grid-relative (1, 0) vector."""
    return np.rad2deg(np.arctan2(-v_north.values, u_east.values))


def test_rotate_vectors_raises_when_required_attrs_missing():
    u, v = _pair([0.0, 1.0], attrs={"GRIB_gridType": "lambert"})
    with pytest.raises(AttributeError):
        rotate_lambert_conical_vectors(u, v)


def test_no_rotation_at_orientation_longitude():
    u, v = _pair([LOV], u=3.0, v=4.0)
    ue, vn = rotate_lambert_conical_vectors(u, v)
    np.testing.assert_allclose(ue, 3.0)
    np.testing.assert_allclose(vn, 4.0)


def test_tangent_cone_rotates_by_sin_latin_times_dlon_and_preserves_speed():
    u, v = _pair([LOV + 10.0, LOV - 10.0], u=3.0, v=4.0)
    ue, vn = rotate_lambert_conical_vectors(u, v)

    expected = np.sin(np.deg2rad(25.0)) * np.array([10.0, -10.0])
    unit_ue, unit_vn = rotate_lambert_conical_vectors(*_pair([LOV + 10.0, LOV - 10.0]))
    np.testing.assert_allclose(_rotation_deg(unit_ue, unit_vn), expected)
    np.testing.assert_allclose(np.hypot(ue, vn), 5.0)


def test_secant_cone_constant_lies_between_standard_parallel_sines():
    ue, vn = rotate_lambert_conical_vectors(*_pair([LOV + 10.0], attrs=_attrs(25.0, 40.0)))
    cone = _rotation_deg(ue, vn).item() / 10.0
    assert np.sin(np.deg2rad(25.0)) < cone < np.sin(np.deg2rad(40.0))


def test_longitude_convention_does_not_change_rotation():
    # same physical longitudes expressed in [0, 360) and [-180, 180)
    ue_360, vn_360 = rotate_lambert_conical_vectors(*_pair([LOV + 10.0, LOV - 30.0]))
    ue_180, vn_180 = rotate_lambert_conical_vectors(*_pair([LOV + 10.0 - 360.0, LOV - 30.0 - 360.0]))
    np.testing.assert_allclose(ue_180, ue_360)
    np.testing.assert_allclose(vn_180, vn_360)


def test_rotate_vectors_wrapper_matches_direct_call_and_marks_earth_relative():
    u, v = _pair([LOV + 10.0])
    expected_u, expected_v = rotate_lambert_conical_vectors(u, v)

    out = rotate_vectors(xr.Dataset({"u": u, "v": v}), vector_pairs=[["u", "v"]])

    xr.testing.assert_allclose(out["u"], expected_u)
    xr.testing.assert_allclose(out["v"], expected_v)
    assert out["u"].attrs["GRIB_uvRelativeToGrid"] == 0
    assert out["v"].attrs["GRIB_uvRelativeToGrid"] == 0
