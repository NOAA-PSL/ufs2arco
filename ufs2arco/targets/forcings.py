import logging

import numpy as np
import xarray as xr
import pandas as pd

logger = logging.getLogger("ufs2arco")

def get_mappings() -> dict:
    return {
        "latitude": _latitude,
        "cos_latitude": _cos_latitude,
        "sin_latitude": _sin_latitude,
        "longitude": _longitude,
        "cos_longitude": _cos_longitude,
        "sin_longitude": _sin_longitude,
        "julian_day": _julian_day,
        "cos_julian_day": _cos_julian_day,
        "sin_julian_day": _sin_julian_day,
        "cos_local_time": _cos_local_time,
        "sin_local_time": _sin_local_time,
        "cos_solar_zenith_angle": _cos_solar_zenith_angle,
        "insolation": _cos_solar_zenith_angle,
    }

def _latitude(xds: xr.Dataset):
    return xds["latitude"]

def _cos_latitude(xds: xr.Dataset):
    return np.cos(xds["latitude"]/180*np.pi)

def _sin_latitude(xds: xr.Dataset):
    return np.sin(xds["latitude"]/180*np.pi)

def _longitude(xds: xr.Dataset):
    return xds["longitude"]

def _cos_longitude(xds: xr.Dataset):
    return np.cos(xds["longitude"]/180*np.pi)

def _sin_longitude(xds: xr.Dataset):
    return np.sin(xds["longitude"]/180*np.pi)

def _julian_day(xds: xr.Dataset):
    ptime = pd.to_datetime(xds["t0"])
    pyears = pd.to_datetime(xds["t0"].dt.year.astype(str))
    delta = ptime - pyears
    jday = delta.days + delta.seconds / 86400
    return xr.DataArray(jday, coords=xds["t0"].coords, attrs={"description": "julian day relative to start of year"})

def _cos_julian_day(xds: xr.Dataset):
    jd = _julian_day(xds)
    cjd = np.cos(jd/365.25*2*np.pi)
    cjd.attrs["description"] = f"cosine of {jd.attrs['description']}"
    return cjd

def _sin_julian_day(xds: xr.Dataset):
    jd = _julian_day(xds)
    sjd = np.sin(jd/365.25*2*np.pi)
    sjd.attrs["description"] = f"sine of {jd.attrs['description']}"
    return sjd

def _local_time(xds: xr.Dataset):
    pd_time = pd.to_datetime(xds["t0"].values)
    delta = pd_time - pd_time.normalize() # gets the delta in hours
    hours = delta.seconds / 86400 * 24 # now this is e.g. 12 at 12z
    hours = xr.DataArray(hours, coords=xds["t0"].coords)
    local_time = (xds["longitude"] / 360 * 24 + hours) % 24
    local_time.attrs["description"] = "relative local time in hours, from time in UTC & longitude"
    return local_time

def _cos_local_time(xds: xr.Dataset):
    lt = _local_time(xds)
    clt = np.cos(lt/24*2*np.pi)
    clt.attrs["description"] = f"cosine of {lt.attrs['description']}"
    return clt

def _sin_local_time(xds: xr.Dataset):
    lt = _local_time(xds)
    slt = np.sin(lt/24*2*np.pi)
    slt.attrs["description"] = f"sin of {lt.attrs['description']}"
    return slt


def _solar_declination_angle(xds: xr.Dataset):
    """Note: this was copied from github.com/ecmwf/earthkit-meteo
    See: src/earthkit/meteo/solar/array/solar.py "solar_declination_angle"

    Returns:
        declination (xr.DataArray): in radians, not in degrees as in earthkit-meteo
        time_correction (xr.DataArray): function of dataset time array
    """
    angle = _julian_day(xds) / 365.25 * np.pi * 2
    declination = np.deg2rad(
        0.396372
        - 22.91327 * np.cos(angle)
        + 4.025430 * np.sin(angle)
        - 0.387205 * np.cos(2 * angle)
        + 0.051967 * np.sin(2 * angle)
        - 0.154527 * np.cos(3 * angle)
        + 0.084798 * np.sin(3 * angle)
    )
    # time correction in [ h.degrees ]
    time_correction = (
        0.004297
        + 0.107029 * np.cos(angle)
        - 1.837877 * np.sin(angle)
        - 0.837378 * np.cos(2 * angle)
        - 2.340475 * np.sin(2 * angle)
    )
    return declination, time_correction

def _cos_solar_zenith_angle(xds: xr.Dataset):
    """Note: this was copied from github.com/ecmwf/earthkit-meteo
    See: src/earthkit/meteo/solar/array/solar.py "cos_solar_zenith_angle"
    """

    declination, time_correction = _solar_declination_angle(xds)
    rlat = np.deg2rad(xds["latitude"])

    sindec_sinlat = np.sin(declination) * np.sin(rlat)
    cosdec_coslat = np.cos(declination) * np.cos(rlat)

    # solar hour angle [h.deg]
    solar_angle = np.deg2rad(
        (xds["t0"].dt.hour - 12) * 15 + xds["longitude"] + time_correction
    )
    zenith_angle = sindec_sinlat + cosdec_coslat * np.cos(solar_angle)
    zenith_angle = zenith_angle.where(zenith_angle > 0., 0.)
    zenith_angle.attrs["description"] = "cosine of solar zenith angle"
    zenith_angle.attrs["attribution"] = "this is computed exactly as in github.com/ecmwf/earthkit-meteo (see src/earthkit/meteo/solar/array/solar.py)"
    return zenith_angle
