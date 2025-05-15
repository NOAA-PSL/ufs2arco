import os
import logging
from typing import Optional

import numpy as np
import dask
import pandas as pd
import xarray as xr

import xesmf
import cf_xarray as cfxr

from ufs2arco.regrid.gaussian_grid import gaussian_latitudes

logger = logging.getLogger("ufs2arco")

def horizontal_regrid(
    xds: xr.Dataset,
    target_grid_path: str,
    regridder_kwargs: dict,
    open_target_kwargs: Optional[dict] = None,
    source_is_on_gaussian_grid: bool = False,
):

    kw = {} if open_target_kwargs is None else open_target_kwargs
    ds_out = xr.open_dataset(os.path.expandvars(target_grid_path), **kw)

    # required for xesmf
    xds = xds.rename({"longitude": "lon", "latitude": "lat"})

    # for the first time, we have to compute the regridder weights no matter what
    # so, figure out if we have the file or not
    kw = regridder_kwargs.copy()
    filename = kw.get(
        "filename",
        f"{kw['method']}_{len(xds.lat)}x{len(xds.lon)}_{len(ds_out.lat)}x{len(ds_out.lon)}.nc",
    )
    filename = os.path.expandvars(filename)
    kw["filename"] = filename
    if os.path.isfile(filename):
        kw["reuse_weights"] = kw.get("reuse_weights", True)
    else:
        kw["reuse_weights"] = False
        logger.info(f"ufs2arco.transforms.horizontal_regrid: couldn't find xesmf weights filename {filename}, they'll be computed now.")

    # check if bounds are there
    if "lat_b" not in xds and "lon_b" not in xds:
        if not os.path.isfile(filename):
            logger.info(f"ufs2arco.transforms.horizontal_regrid: did not find 'lat_b' or 'lon_b' in source, computing bounds.")
        xds = get_bounds(xds, is_gaussian=source_is_on_gaussian_grid)
        if not os.path.isfile(filename):
            logger.info(f"ufs2arco.transforms.horizontal_regrid: computed bounds are\nLongitude\n{xds.lon_b}\nLatitude\n{xds.lat_b}")

    # do the work
    regridder = xesmf.Regridder(
        ds_in=xds,
        ds_out=ds_out,
        **kw,
    )
    ds_out = regridder(xds, keep_attrs=True)

    # ds_out has the lat/lon boundaries from input dataset
    # remove these because it doesn't make sense anymore
    ds_out = ds_out.drop_vars(["lat_b", "lon_b"])

    # this one is a weird one, created by xesmf's global grid util creator
    if "latitude_longitude" in ds_out:
        if np.isnan(ds_out["latitude_longitude"]):
            ds_out = ds_out.drop_vars("latitude_longitude")
    ds_out = ds_out.rename({"lon": "longitude", "lat": "latitude",})
    return ds_out

def get_bounds(xds, is_gaussian=False):
    """
    Use cf_xarray to get the bounds of a rectilinear grid
    """
    xds = xds.cf.add_bounds(["lat", "lon"])

    for key in ["lat", "lon"]:
        corners = cfxr.bounds_to_vertices(
            bounds=xds[f"{key}_bounds"],
            bounds_dim="bounds",
            order=None,
        )
        xds = xds.assign_coords({f"{key}_b": corners})
        xds = xds.drop_vars(f"{key}_bounds")

    if is_gaussian:
        xds = xds.drop_vars("lat_b")
        _, lat_b = gaussian_latitudes(len(xds.lat)//2)
        lat_b = np.concatenate([lat_b[:,0], [lat_b[-1,-1]]])
        if xds["lat"][0] > 0:
            lat_b = lat_b[::-1]
        xds["lat_b"] = xr.DataArray(
            lat_b,
            dims="lat_vertices",
        )
        xds = xds.set_coords("lat_b")
    return xds


def create_output_dataset(lat, lon, is_gaussian):
    xds = xr.Dataset({
        "lat": lat,
        "lon": lon,
    })
    return get_bounds(xds, is_gaussian)
