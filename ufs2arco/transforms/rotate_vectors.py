"""
Rotate vectors, e.g. from model grid orientation to eastward/northward facing
"""

import numpy as np
import xarray as xr

def rotate_vectors(
    xds: xr.Dataset,
    vector_pairs: list[list[str, str]],
):

    for (u, v) in vector_pairs:
        xds[u], xds[v] = rotate_lambert_conical_vectors(xds[u], xds[v])

    return xds

def rotate_lambert_conical_vectors(u_grid, v_grid):
    """Note that this mirrors the calculations performed by NCEPLIBS-ip (often accessed via wgrib2)
    here: https://github.com/NOAA-EMC/NCEPLIBS-ip/blob/develop/src/ip_lambert_conf_grid_mod.F90

    See some verification of this here: https://gist.github.com/timothyas/4a0ea5a36b3015fd756a150be60d250f

    Args:
        u_grid, v_grid (xr.DataArray): the u, v vector pair, with data relative to model grid

    Returns:
        u_east, v_north (xr.DataArray): the u, v vector pair, with data pointing eastward and northward, respectively
    """
    uname = u_grid.name
    vname = v_grid.name
    not_here = []
    for key in [
        "GRIB_gridType",
        "GRIB_uvRelativeToGrid",
        "GRIB_latitudeOfFirstGridPointInDegrees",
        "GRIB_longitudeOfFirstGridPointInDegrees",
        "GRIB_Latin1InDegrees",
        "GRIB_Latin2InDegrees",
        "GRIB_LoVInDegrees",
    ]:
        try:
            assert key in u_grid.attrs
            assert key in v_grid.attrs
            assert u_grid.attrs[key] == v_grid.attrs[key]
        except:
            not_here.append(key)

    if len(not_here) > 0:
        raise AttributeError(f"rotate_lambert_conical_vectors: could not find the following GRIB attributes for the variable ({u_grid.name}, {v_grid.name}), or it's inconsistent between the two: {not_here}")

    assert u_grid.attrs["GRIB_gridType"] == "lambert", f"rotate_lambert_conical_vectors: only lambert conical rotation implemented, not true for ({u_grid.name}, {v_grid.name})"
    assert u_grid.attrs["GRIB_uvRelativeToGrid"] == 1, f"rotate_lambert_conical_vectors: only implemented for rotating from grid orientation to east/north, not true for ({u_grid.name}, {v_grid.name})"

    rlat1 = u_grid.attrs["GRIB_latitudeOfFirstGridPointInDegrees"]
    rlon1 = u_grid.attrs["GRIB_longitudeOfFirstGridPointInDegrees"]
    rlati1 = u_grid.attrs["GRIB_Latin1InDegrees"]
    rlati2 = u_grid.attrs["GRIB_Latin2InDegrees"]
    orient = u_grid.attrs["GRIB_LoVInDegrees"]


    # now, do the actual work
    eps = np.finfo(float).eps

    if np.abs(rlati1 - rlati2) < eps:
        angle = np.sin(np.deg2rad(rlati1))

    else:
        angle = np.log(
            np.cos(np.deg2rad(rlati1))/np.cos(np.deg2rad(rlati2))
        ) / np.log(
            np.tan(np.deg2rad((90-rlati1)/2))/np.tan(np.deg2rad((90-rlati2)/2))
        )

    dlon = (u_grid["longitude"] - orient + 180 + 3600 % 360) - 180

    crot = np.cos(angle*np.deg2rad(dlon)).astype(u_grid.dtype)
    srot = np.sin(angle*np.deg2rad(dlon)).astype(u_grid.dtype)

    with xr.set_options(keep_attrs=True):
        u_east  = u_grid * crot + v_grid * srot
        v_north = v_grid * crot - u_grid * srot

    # manage coords
    u_east = u_east.rename(uname)
    v_north = v_north.rename(vname)
    u_east.attrs["GRIB_uvRelativeToGrid"] = 0
    v_north.attrs["GRIB_uvRelativeToGrid"] = 0
    u_east.attrs["GRIB_latitudeOfSouthernPoleInDegrees"] = -90.0
    v_north.attrs["GRIB_latitudeOfSouthernPoleInDegrees"] = -90.0

    u_east.attrs["rotation"] = "rotated to point eastward via ufs2arco.transforms.rotate_lambert_conical_vectors"
    v_north.attrs["rotation"] = "rotated to point northward via ufs2arco.transforms.rotate_lambert_conical_vectors"
    return u_east, v_north
