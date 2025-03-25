import logging

import numpy as np
import xarray as xr
import pandas as pd

logger = logging.getLogger("ufs2arco")

def get_mappings(time="time") -> dict:
    return {
        "latitude": _latitude,
        "cos_latitude": _cos_latitude,
        "sin_latitude": _sin_latitude,
        "longitude": _longitude,
        "cos_longitude": _cos_longitude,
        "sin_longitude": _sin_longitude,
        "julian_day": lambda xds: _julian_day(xds, time=time),
        "cos_julian_day": lambda xds: _cos_julian_day(xds, time=time),
        "sin_julian_day": lambda xds: _sin_julian_day(xds, time=time),
        "cos_local_time": lambda xds: _cos_local_time(xds, time=time),
        "sin_local_time": lambda xds: _sin_local_time(xds, time=time),
        "cos_solar_zenith_angle": lambda xds: _cos_solar_zenith_angle(xds, time=time),
        "insolation": lambda xds: _cos_solar_zenith_angle(xds, time=time),
    }

def _latitude(xds: xr.Dataset):
    lat = xds["latitude"]
    lat.attrs["computed_forcing"] = True
    lat.attrs["constant_in_time"] = True
    return lat

def _cos_latitude(xds: xr.Dataset):
    clat = np.cos(xds["latitude"]/180*np.pi)
    clat.attrs["computed_forcing"] = True
    clat.attrs["constant_in_time"] = True
    return clat

def _sin_latitude(xds: xr.Dataset):
    slat = np.sin(xds["latitude"]/180*np.pi)
    slat.attrs["computed_forcing"] = True
    slat.attrs["constant_in_time"] = True
    return slat

def _longitude(xds: xr.Dataset):
    lon = xds["longitude"]
    lon.attrs["computed_forcing"] = True
    lon.attrs["constant_in_time"] = True
    return lon

def _cos_longitude(xds: xr.Dataset):
    clon = np.cos(xds["longitude"]/180*np.pi)
    clon.attrs["computed_forcing"] = True
    clon.attrs["constant_in_time"] = True
    return clon

def _sin_longitude(xds: xr.Dataset):
    slon = np.sin(xds["longitude"]/180*np.pi)
    slon.attrs["computed_forcing"] = True
    slon.attrs["constant_in_time"] = True
    return slon

def _julian_day(xds: xr.Dataset, time="time"):
    ptime = pd.to_datetime(xds[time])
    pyears = pd.to_datetime(xds[time].dt.year.astype(str))
    delta = ptime - pyears
    jday = delta.days + delta.seconds / 86400
    jday = xr.DataArray(
        jday,
        coords=xds[time].coords,
        attrs={
            "description": "julian day relative to start of year",
            "computed_forcing": True,
            "constant_in_time": False,
        },
    )
    return jday

def _cos_julian_day(xds: xr.Dataset, time="time"):
    jd = _julian_day(xds, time=time)
    cjd = np.cos(jd/365.25*2*np.pi)
    cjd.attrs["description"] = f"cosine of {jd.attrs['description']}"
    cjd.attrs["computed_forcing"] = True
    cjd.attrs["constant_in_time"] = False
    return cjd

def _sin_julian_day(xds: xr.Dataset, time="time"):
    jd = _julian_day(xds, time=time)
    sjd = np.sin(jd/365.25*2*np.pi)
    sjd.attrs["description"] = f"sine of {jd.attrs['description']}"
    sjd.attrs["computed_forcing"] = True
    sjd.attrs["constant_in_time"] = False
    return sjd

def _local_time(xds: xr.Dataset, time="time"):
    pd_time = pd.to_datetime(xds[time].values)
    delta = pd_time - pd_time.normalize() # gets the delta in hours
    hours = delta.seconds / 86400 * 24 # now this is e.g. 12 at 12z
    hours = xr.DataArray(hours, coords=xds[time].coords)
    local_time = (xds["longitude"] / 360 * 24 + hours) % 24
    local_time.attrs["description"] = "relative local time in hours, from time in UTC & longitude"
    local_time.attrs["computed_forcing"] = True
    local_time.attrs["constant_in_time"] = False
    return local_time

def _cos_local_time(xds: xr.Dataset, time="time"):
    lt = _local_time(xds, time=time)
    clt = np.cos(lt/24*2*np.pi)
    clt.attrs["description"] = f"cosine of {lt.attrs['description']}"
    clt.attrs["computed_forcing"] = True
    clt.attrs["constant_in_time"] = False
    return clt

def _sin_local_time(xds: xr.Dataset, time="time"):
    lt = _local_time(xds, time=time)
    slt = np.sin(lt/24*2*np.pi)
    slt.attrs["description"] = f"sin of {lt.attrs['description']}"
    slt.attrs["computed_forcing"] = True
    slt.attrs["constant_in_time"] = False
    return slt

def _solar_declination_angle(xds: xr.Dataset, time="time"):
    """Note: this was copied from github.com/ecmwf/earthkit-meteo
    See: src/earthkit/meteo/solar/array/solar.py "solar_declination_angle"

    Returns:
        declination (xr.DataArray): in radians, not in degrees as in earthkit-meteo
        time_correction (xr.DataArray): function of dataset time array
    """
    angle = _julian_day(xds, time=time) / 365.25 * np.pi * 2
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

def _cos_solar_zenith_angle(xds: xr.Dataset, time="time"):
    """Note: this was copied from github.com/ecmwf/earthkit-meteo
    See: src/earthkit/meteo/solar/array/solar.py "cos_solar_zenith_angle"
    """

    declination, time_correction = _solar_declination_angle(xds, time=time)
    rlat = np.deg2rad(xds["latitude"])

    sindec_sinlat = np.sin(declination) * np.sin(rlat)
    cosdec_coslat = np.cos(declination) * np.cos(rlat)

    # solar hour angle [h.deg]
    solar_angle = np.deg2rad(
        (xds[time].dt.hour - 12) * 15 + xds["longitude"] + time_correction
    )
    zenith_angle = sindec_sinlat + cosdec_coslat * np.cos(solar_angle)
    zenith_angle = zenith_angle.where(zenith_angle > 0., 0.)
    zenith_angle.attrs["description"] = "cosine of solar zenith angle"
    zenith_angle.attrs["attribution"] = "this is computed exactly as in github.com/ecmwf/earthkit-meteo (see src/earthkit/meteo/solar/array/solar.py)"
    zenith_angle.attrs["computed_forcing"] = True
    zenith_angle.attrs["constant_in_time"] = False
    return zenith_angle
