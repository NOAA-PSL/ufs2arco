import numpy as np
import pandas as pd
import xarray as xr

from ufs2arco.targets import forcings


def _dataset():
    return xr.Dataset(
        coords={
            "time": pd.date_range("2020-01-01T00", periods=2, freq="12h"),
            "longitude": ("longitude", np.array([-180.0, 0.0, 180.0])),
            "latitude": ("latitude", np.array([-30.0, 30.0])),
        }
    )


def test_day_progress_is_bounded_and_float32():
    xds = _dataset()
    dp = forcings._day_progress(xds, time="time")
    assert dp.dtype == np.float32
    assert float(dp.min()) >= 0.0
    assert float(dp.max()) < 1.0


def test_insolation_alias_matches_cos_solar_zenith_angle():
    xds = _dataset()
    mappings = forcings.get_mappings(time="time")
    a = mappings["insolation"](xds)
    b = mappings["cos_solar_zenith_angle"](xds)
    np.testing.assert_allclose(a.values, b.values)


def test_get_mappings_contains_expected_core_keys():
    mappings = forcings.get_mappings(time="time")
    for key in ["latitude", "cos_latitude", "julian_day", "cos_local_time", "insolation", "cos_year_progress"]:
        assert key in mappings


def test_latitude_and_longitude_forcing_attrs():
    xds = _dataset()
    lat = forcings._latitude(xds)
    lon = forcings._longitude(xds)
    assert lat.attrs["computed_forcing"] is True
    assert lat.attrs["constant_in_time"] is True
    assert lon.attrs["computed_forcing"] is True
    assert lon.attrs["constant_in_time"] is True


def test_julian_day_progression_and_attrs():
    xds = _dataset()
    jd = forcings._julian_day(xds, time="time")
    assert jd.attrs["computed_forcing"] is True
    assert jd.attrs["constant_in_time"] is False
    assert float(jd.isel(time=1).values) > float(jd.isel(time=0).values)


def test_year_progress_dtype_and_bounds():
    xds = _dataset()
    yp = forcings._year_progress(xds, time="time")
    assert yp.dtype == np.float32
    assert float(yp.min()) >= 0.0
    assert float(yp.max()) < 1.0
