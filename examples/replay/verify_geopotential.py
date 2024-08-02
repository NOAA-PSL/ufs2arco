import logging
import sys
import numpy as np
import dask
import xarray as xr
import xesmf
import cf_xarray as cfxr

from ufs2arco.regrid.gaussian_grid import gaussian_latitudes
from ufs2arco import Layers2Pressure

class SimpleFormatter(logging.Formatter):
    def format(self, record):
        record.relativeCreated = record.relativeCreated // 1000
        return super().format(record)

def setup_log(level=logging.INFO):

    logging.basicConfig(
        stream=sys.stdout,
        level=level,
    )
    logger = logging.getLogger()
    formatter = SimpleFormatter(fmt="[%(relativeCreated)d s] [%(levelname)s] %(message)s")
    for handler in logger.handlers:
        handler.setFormatter(formatter)

def open_datasets():
    """
    Returns:
        predictions, targets, original_target_dataset, independent_truth_dataset
    """

    rds = xr.open_zarr(
        "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree/03h-freq/zarr/fv3.zarr",
        storage_options={"anon":True},
    )
    rds = rds.drop_vars(["cftime", "ftime"])
    rds = rds.where(rds.time.dt.hour / 6 == rds.time.dt.hour // 6, drop=True)
    rds = rds.isel(time=np.random.randint(low=0,high=len(rds.time), size=20))

    rds = rds[["pressfc", "hgtsfc", "tmp", "spfh", "delz"]]
    logging.info(f"Loading {rds.nbytes / 1e9} GB dataset...")
    rds = rds.load()
    logging.info(" ... done")
    rds = rds.sortby("time")

    truth = xr.open_zarr(
        "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
        storage_options={"token":"anon"},
    )

    # subsample in space to avoid poles
    truth = truth.sel(latitude=slice(-89, 89))

    rds = rds.rename({"pfull": "level", "grid_xt": "lon", "grid_yt": "lat"})
    return rds, truth


def get_bounds(xds, is_gaussian=False):
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


def regrid_and_rename(xds, truth):
    """Note that it's assumed the truth dataset is not on a Gaussian grid but input is"""

    ds_out = create_output_dataset(
        lat=truth["latitude"].values,
        lon=truth["longitude"].values,
        is_gaussian=False,
    )
    if "lat_b" not in xds and "lon_b" not in xds:
        xds = get_bounds(xds, is_gaussian=True)

    regridder = xesmf.Regridder(
        ds_in=xds,
        ds_out=ds_out,
        method="conservative",
        reuse_weights=False,
    )
    ds_out = regridder(xds, keep_attrs=True)

    rename_dict = {
        "pressfc": "surface_pressure",
        "ugrd10m": "10m_u_component_of_wind",
        "vgrd10m": "10m_v_component_of_wind",
        "tmp2m": "2m_temperature",
        "tmp": "temperature",
        "ugrd": "u_component_of_wind",
        "vgrd": "v_component_of_wind",
        "dzdt": "vertical_velocity",
        "spfh": "specific_humidity",
        "prateb_ave": "total_precipitation_3hr",
        "lat": "latitude",
        "lon": "longitude",
    }
    rename_dict = {k: v for k,v in rename_dict.items() if k in ds_out}
    ds_out = ds_out.rename(rename_dict)

    # ds_out has the lat/lon boundaries from input dataset
    # remove these because it doesn't make sense anymore
    ds_out = ds_out.drop_vars(["lat_b", "lon_b"])
    return ds_out


def interp2pressure(xds, plevels, hgtsfc=None):
    """Assume plevels is in hPa"""

    lp = Layers2Pressure()
    prsl = lp.calc_layer_mean_pressure(xds["pressfc"], xds["tmp"], xds["spfh"], xds["delz"])

    if "geopotential" not in xds and hgtsfc is not None:
        xds["geopotential"] = lp.calc_geopotential(hgtsfc, xds["delz"])
        xds["geopotential"] = xds["geopotential"].chunk({
            "time": 1,
            "level": 1,
            "lat": -1,
            "lon": -1,
        })

    vars3d = ["geopotential"]
    pds = xr.Dataset()
    plevels = np.array(list(plevels))
    pds["level"] = xr.DataArray(
        plevels,
        coords={"level": plevels},
        dims=("level",),
        attrs={
            "description": "Pressure level",
            "units": "hPa",
        },
    )
    results = {k: list() for k in vars3d}
    for p in plevels:


        cds = lp.get_interp_coefficients(p*100, prsl)
        mask = (cds["is_right"].sum("level") > 0) & (cds["is_left"].sum("level") > 0)
        for key in vars3d:
            logging.info(f"Interpolating {key} to {p} hPa")
            interpolated = lp.interp2pressure(xds[key], p*100, prsl, cds)
            interpolated = interpolated.expand_dims({"level": [p]})
            interpolated = interpolated.where(mask)
            results[key].append(interpolated.compute())

    for key in vars3d:
        pds[key] = xr.concat(results[key], dim="level")

    return pds


if __name__ == "__main__":

    setup_log()
    dask.config.set(scheduler="threads", num_workers=48)

    rds, truth = open_datasets()
    hgtsfc = rds["hgtsfc"].isel(time=0).drop_vars(["time"])
    hgtsfc = hgtsfc.load()
    rds["hgtsfc"] = hgtsfc

    rds = interp2pressure(rds, [100, 500, 850], hgtsfc)
    logging.info(f"Interpolated to pressure levels...")

    # regrid and rename variables
    rds = regrid_and_rename(rds, truth)
    logging.info(f"Done regridding...")

    path = f"/p1-evaluation/v1/validation/replay.geopotential.20.zarr"
    rds.to_zarr(path)
    logging.info(f"Done writing to {path}")
