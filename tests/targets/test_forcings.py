import numpy as np
import pandas as pd
import pytest
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


def test_latitude_and_longitude_forcing_attrs():
    xds = _dataset()
    lat = forcings._latitude(xds)
    lon = forcings._longitude(xds)
    assert lat.attrs["computed_forcing"] is True
    assert lat.attrs["constant_in_time"] is True
    assert lon.attrs["computed_forcing"] is True
    assert lon.attrs["constant_in_time"] is True


def test_year_progress_dtype_and_bounds():
    xds = _dataset()
    yp = forcings._year_progress(xds, time="time")
    assert yp.dtype == np.float32
    assert float(yp.min()) >= 0.0
    assert float(yp.max()) < 1.0


def _grid_dataset(times=None):
    if times is None:
        times = pd.date_range("2020-01-01T00", periods=2, freq="12h")
    return xr.Dataset(
        coords={
            "time": times,
            "longitude": ("longitude", np.array([0.0, 90.0, 180.0, 270.0])),
            "latitude": ("latitude", np.array([-45.0, 0.0, 45.0])),
        }
    )


def test_cos_sin_latitude_exact_values():
    xds = _grid_dataset()  # latitude = [-45, 0, 45] degrees
    clat = forcings._cos_latitude(xds)
    slat = forcings._sin_latitude(xds)
    half_sqrt2 = np.sqrt(2) / 2
    np.testing.assert_allclose(clat.values, [half_sqrt2, 1.0, half_sqrt2], atol=1e-12)
    np.testing.assert_allclose(slat.values, [-half_sqrt2, 0.0, half_sqrt2], atol=1e-12)
    for da in (clat, slat):
        assert da.attrs["computed_forcing"] is True
        assert da.attrs["constant_in_time"] is True


def test_cos_sin_longitude_exact_values():
    xds = _grid_dataset()
    clon = forcings._cos_longitude(xds)
    slon = forcings._sin_longitude(xds)
    np.testing.assert_allclose(clon.values, np.cos(np.deg2rad(xds["longitude"].values)), atol=1e-12)
    np.testing.assert_allclose(slon.values, np.sin(np.deg2rad(xds["longitude"].values)), atol=1e-12)


def test_cos_sin_julian_day_match_manual():
    xds = _grid_dataset()
    jd = forcings._julian_day(xds, time="time")
    cjd = forcings._cos_julian_day(xds, time="time")
    sjd = forcings._sin_julian_day(xds, time="time")
    np.testing.assert_allclose(cjd.values, np.cos(jd.values / 365.25 * 2 * np.pi), atol=1e-12)
    np.testing.assert_allclose(sjd.values, np.sin(jd.values / 365.25 * 2 * np.pi), atol=1e-12)
    assert "cosine" in cjd.attrs["description"]
    assert "sine" in sjd.attrs["description"]


def test_julian_day_starts_at_zero_and_increases():
    jd = forcings._julian_day(_grid_dataset(), time="time")  # starts 2020-01-01T00
    assert float(jd.isel(time=0)) == 0.0
    assert float(jd.isel(time=1)) > 0.0
    assert jd.attrs["computed_forcing"] is True
    assert jd.attrs["constant_in_time"] is False


def test_local_time_offset_by_longitude():
    xds = _grid_dataset(times=pd.DatetimeIndex([pd.Timestamp("2020-01-01T00")]))
    lt = forcings._local_time(xds, time="time").isel(time=0)
    expected = (xds["longitude"].values / 360 * 24) % 24
    np.testing.assert_allclose(lt.values, expected, atol=1e-9)
    assert float(lt.max()) < 24.0
    assert float(lt.min()) >= 0.0


def test_cos_sin_local_time_identity():
    xds = _grid_dataset()
    clt = forcings._cos_local_time(xds, time="time")
    slt = forcings._sin_local_time(xds, time="time")
    np.testing.assert_allclose((clt**2 + slt**2).values, 1.0, atol=1e-9)


@pytest.mark.parametrize(
    "time,declination_deg",
    [("2020-03-20T12", 0.0), ("2020-06-20T12", 23.44), ("2020-12-21T12", -23.44)],
)
def test_cos_solar_zenith_angle_is_cos_lat_minus_declination_at_noon_and_zero_at_midnight(
    time, declination_deg
):
    # at 12Z it is local noon at longitude 0 and local midnight at longitude 180
    xds = xr.Dataset(
        coords={
            "time": pd.DatetimeIndex([time]),
            "longitude": ("longitude", np.array([0.0, 180.0])),
            "latitude": ("latitude", np.array([-45.0, 0.0, 45.0])),
        }
    )
    csza = forcings._cos_solar_zenith_angle(xds, time="time").isel(time=0)

    expected_noon = np.cos(np.deg2rad(xds["latitude"].values - declination_deg))
    np.testing.assert_allclose(csza.sel(longitude=0.0), expected_noon, atol=0.01)
    np.testing.assert_array_equal(csza.sel(longitude=180.0), 0.0)
    assert csza.attrs["computed_forcing"] is True
    assert csza.attrs["constant_in_time"] is False


def test_cos_sin_year_progress_identity_and_dtype():
    xds = _grid_dataset()
    cyp = forcings._cos_year_progress(xds, time="time")
    syp = forcings._sin_year_progress(xds, time="time")
    np.testing.assert_allclose((cyp.astype(np.float64)**2 + syp.astype(np.float64)**2).values, 1.0, atol=1e-6)


def test_cos_sin_day_progress_identity_and_longitude_dependence():
    xds = _grid_dataset()
    cdp = forcings._cos_day_progress(xds, time="time")
    sdp = forcings._sin_day_progress(xds, time="time")
    np.testing.assert_allclose((cdp.astype(np.float64)**2 + sdp.astype(np.float64)**2).values, 1.0, atol=1e-6)
    # day progress varies with longitude at a fixed time
    dp = forcings._day_progress(xds, time="time").isel(time=0)
    assert len(np.unique(np.round(dp.values, 6))) > 1


@pytest.mark.parametrize(
    "time,expected_deg",
    [("2020-01-01T00", -23.0), ("2020-03-20T12", 0.0), ("2020-06-20T12", 23.44)],
)
def test_solar_declination_angle_matches_seasonal_values(time, expected_deg):
    xds = _grid_dataset(times=pd.DatetimeIndex([time]))
    declination, _ = forcings._solar_declination_angle(xds, time="time")
    assert np.rad2deg(declination.item()) == pytest.approx(expected_deg, abs=0.5)
